import argparse
import datetime as dt
import os
import queue
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk

import numpy as np
import sounddevice as sd
from deep_translator import GoogleTranslator
from faster_whisper import WhisperModel


@dataclass
class AppConfig:
    sample_rate: int = 16000
    channels: int = 1
    chunk_seconds: float = 4.0
    model_name: str = "small.en"
    compute_type: str = "int8"
    language: str = "en"
    min_text_len: int = 3


def choose_default_device(preferred_index: int | None = None) -> tuple[int | None, str]:
    devices = sd.query_devices()
    if preferred_index is not None:
        return preferred_index, f"выбрано вручную устройство #{preferred_index}"

    for idx, dev in enumerate(devices):
        if dev.get("max_input_channels", 0) > 0 and "blackhole" in dev["name"].lower():
            return idx, f"автовыбор BlackHole: #{idx} {dev['name']}"

    return None, "используется системное input-устройство по умолчанию"


def get_blackhole_inputs() -> list[tuple[int, str]]:
    devices = sd.query_devices()
    found: list[tuple[int, str]] = []
    for idx, dev in enumerate(devices):
        if dev.get("max_input_channels", 0) > 0 and "blackhole" in dev["name"].lower():
            found.append((idx, dev["name"]))
    return found


def print_doctor() -> int:
    print("VoiceTalk doctor")
    print("================")
    print(f"Python: {os.sys.version.split()[0]}")
    try:
        import tkinter  # noqa: F401
        print("Tkinter: OK")
    except Exception as exc:
        print(f"Tkinter: FAIL ({exc})")
        return 1

    blackholes = get_blackhole_inputs()
    if blackholes:
        print("BlackHole input: OK")
        for idx, name in blackholes:
            print(f"  - [{idx}] {name}")
    else:
        print("BlackHole input: NOT FOUND")
        print("  Install and reboot:")
        print("  brew install --cask blackhole-2ch")
        print("  Then reboot macOS")

    print("Tip: run `python3 main.py --list-devices` to verify audio inputs.")
    return 0


class TranslatorApp:
    def __init__(self, root: tk.Tk, device_index: int | None, cfg: AppConfig, subtitles_path: str | None):
        self.root = root
        self.cfg = cfg
        self.device_index = device_index
        self.raw_audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self.audio_chunk_queue: queue.Queue[np.ndarray] = queue.Queue()
        self.text_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self.stop_event = threading.Event()
        self.capture_thread: threading.Thread | None = None
        self.worker_thread: threading.Thread | None = None
        self.stream: sd.InputStream | None = None
        self.subtitles_path = subtitles_path

        self.model: WhisperModel | None = None
        self.translator = GoogleTranslator(source="en", target="ru")
        self.model_ready = threading.Event()

        self.status_var = tk.StringVar(value="Загрузка модели распознавания...")
        self.topmost_var = tk.BooleanVar(value=True)
        self.running = False
        self.toggle_btn: ttk.Button | None = None
        self._build_ui()
        self.root.attributes("-topmost", True)
        self.root.after(50, self._bring_to_front)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(200, self._drain_text_queue)
        threading.Thread(target=self._load_model, daemon=True).start()

    def _bring_to_front(self) -> None:
        # Helps on macOS when app opens without focused window.
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()

    def _load_model(self) -> None:
        try:
            self.model = WhisperModel(self.cfg.model_name, device="auto", compute_type=self.cfg.compute_type)
            self.model_ready.set()
            self.status_var.set("Готово к запуску")
        except Exception as exc:
            self.status_var.set(f"Ошибка загрузки модели: {exc}")

    def _build_ui(self) -> None:
        self.root.title("VoiceTalk: EN -> RU")
        self.root.geometry("960x560")

        top = ttk.Frame(self.root, padding=10)
        top.pack(fill="x")

        self.toggle_btn = ttk.Button(top, text="Старт", command=self.toggle_run)
        self.toggle_btn.pack(side="left", padx=(0, 8))
        ttk.Button(top, text="Сброс", command=self.clear_texts).pack(side="left", padx=(8, 0))
        ttk.Checkbutton(
            top,
            text="Поверх окон",
            variable=self.topmost_var,
            command=self._toggle_topmost,
        ).pack(side="left", padx=10)
        ttk.Label(top, textvariable=self.status_var).pack(side="left", padx=12)

        body = ttk.Frame(self.root, padding=10)
        body.pack(fill="both", expand=True)

        ttk.Label(body, text="Распознанный английский:").pack(anchor="w")
        self.en_text = tk.Text(body, wrap="word", height=14)
        self.en_text.pack(fill="both", expand=True, pady=(4, 10))

        ttk.Label(body, text="Перевод на русский:").pack(anchor="w")
        self.ru_text = tk.Text(body, wrap="word", height=14)
        self.ru_text.pack(fill="both", expand=True, pady=(4, 0))

    def _toggle_topmost(self) -> None:
        self.root.attributes("-topmost", bool(self.topmost_var.get()))

    def start(self) -> None:
        if not self.model_ready.is_set():
            self.status_var.set("Модель еще загружается, подожди 5-20 секунд...")
            return
        if self.capture_thread and self.capture_thread.is_alive():
            return

        self.stop_event.clear()
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.worker_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.capture_thread.start()
        self.worker_thread.start()
        self.running = True
        self._update_toggle_button()
        device_msg = f"Слушаю аудио (device={self.device_index if self.device_index is not None else 'default'})"
        self.status_var.set(device_msg)

    def stop(self) -> None:
        self.stop_event.set()
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        self.running = False
        self._update_toggle_button()
        self.status_var.set("Остановлено")

    def toggle_run(self) -> None:
        if self.running:
            self.stop()
        else:
            self.start()

    def _update_toggle_button(self) -> None:
        if self.toggle_btn is not None:
            self.toggle_btn.config(text="Стоп" if self.running else "Старт")

    def clear_texts(self) -> None:
        self.en_text.delete("1.0", "end")
        self.ru_text.delete("1.0", "end")
        if self.subtitles_path:
            try:
                with open(self.subtitles_path, "w", encoding="utf-8"):
                    pass
            except Exception:
                pass
        self.status_var.set("Текст очищен")

    def _on_close(self) -> None:
        self.stop()
        self.root.destroy()

    def _capture_loop(self) -> None:
        chunk_samples = int(self.cfg.sample_rate * self.cfg.chunk_seconds)
        buffer = np.empty((0, self.cfg.channels), dtype=np.float32)

        def callback(indata: np.ndarray, frames: int, _time_info, status) -> None:
            if status:
                self.status_var.set(f"Проблема аудио: {status}")
            self.raw_audio_queue.put(indata.copy())

        try:
            self.stream = sd.InputStream(
                samplerate=self.cfg.sample_rate,
                channels=self.cfg.channels,
                dtype="float32",
                device=self.device_index,
                callback=callback,
            )
            self.stream.start()
        except Exception as exc:
            self.status_var.set(f"Ошибка открытия устройства: {exc}")
            return

        while not self.stop_event.is_set():
            try:
                block = self.raw_audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            buffer = np.concatenate([buffer, block], axis=0)
            if len(buffer) >= chunk_samples:
                audio = buffer[:chunk_samples, 0]
                buffer = buffer[chunk_samples:]
                self.audio_chunk_queue.put(audio.astype(np.float32))

        self.stop_event.set()

    def _process_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                item = self.audio_chunk_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if not isinstance(item, np.ndarray) or item.ndim != 1:
                continue

            max_amp = float(np.max(np.abs(item))) if len(item) else 0.0
            if max_amp < 0.01:
                continue

            try:
                if self.model is None:
                    continue
                segments, _ = self.model.transcribe(
                    item,
                    language=self.cfg.language,
                    vad_filter=True,
                    beam_size=1,
                )
                text_en = " ".join(seg.text.strip() for seg in segments).strip()
                if len(text_en) < self.cfg.min_text_len:
                    continue
                text_ru = self.translator.translate(text_en)
                self.text_queue.put((text_en, text_ru))
                self._append_subtitle(text_en, text_ru)
            except Exception as exc:
                self.status_var.set(f"Ошибка распознавания/перевода: {exc}")
                time.sleep(0.4)

    def _append_subtitle(self, text_en: str, text_ru: str) -> None:
        if not self.subtitles_path:
            return
        try:
            ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            line = f"[{ts}] EN: {text_en}\n[{ts}] RU: {text_ru}\n\n"
            with open(self.subtitles_path, "a", encoding="utf-8") as fh:
                fh.write(line)
        except Exception:
            pass

    def _drain_text_queue(self) -> None:
        updated = False
        while True:
            try:
                text_en, text_ru = self.text_queue.get_nowait()
            except queue.Empty:
                break

            updated = True
            self.en_text.insert("end", text_en + "\n")
            self.ru_text.insert("end", text_ru + "\n")
            self.en_text.see("end")
            self.ru_text.see("end")

        if updated:
            self.status_var.set("Текст обновлён")
        self.root.after(200, self._drain_text_queue)


def list_input_devices() -> None:
    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        if dev.get("max_input_channels", 0) > 0:
            print(f"[{idx}] {dev['name']} (in={dev['max_input_channels']})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live EN->RU translation from audio input.")
    parser.add_argument("--device", type=int, default=None, help="Audio input device index")
    parser.add_argument("--list-devices", action="store_true", help="Show input devices and exit")
    parser.add_argument("--doctor", action="store_true", help="Check environment and audio setup")
    parser.add_argument("--model", default="small.en", help="faster-whisper model name")
    parser.add_argument(
        "--subtitles-file",
        default=None,
        help="Path to save subtitles log (default: subtitles-YYYYMMDD-HHMMSS.txt)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_devices:
        list_input_devices()
        return
    if args.doctor:
        raise SystemExit(print_doctor())

    cfg = AppConfig(model_name=args.model)
    device_index, msg = choose_default_device(args.device)
    subtitles_path = args.subtitles_file
    if not subtitles_path:
        ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        subtitles_path = os.path.abspath(f"subtitles-{ts}.txt")
    print(f"Device: {msg}")
    print(f"Subtitles file: {subtitles_path}")
    root = tk.Tk()
    TranslatorApp(root, device_index, cfg, subtitles_path)
    root.mainloop()


if __name__ == "__main__":
    main()
