from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import threading

# Функция для мониторинга и извлечения текста с первой веб-страницы
def monitor_text():
    global previous_text
    driver1 = webdriver.Chrome()

    # Открыть веб-страницу с формой
    url1 = 'https://www.textfromtospeech.com/ru/voice-to-text/'  # Замените на URL вашей веб-страницы
    driver1.get(url1)

    try:
        while True:
            try:
                # Найти элемент <span> по его id
                span_element = driver1.find_element(By.ID, 'text-box')

                # Извлечь текст из элемента
                if span_element:
                    text = span_element.text
                    # Проверить, изменился ли текст
                    if text != previous_text:
                        print("Извлеченный текст:", text)
                        previous_text = text
                        # Передать текст в окно переводчика
                        insert_text(text)
                else:
                    print("Элемент <span> не найден")
            except Exception as e:
                print("Ошибка при поиске элемента:", e)

            # Подождать некоторое время перед следующей попыткой
            time.sleep(0.5)  # Здесь можно задать интервал в секундах

    except KeyboardInterrupt:
        # Обработка прерывания Ctrl+C
        pass
    finally:
        # Закрыть браузер при завершении потока
        driver1.quit()

# Функция для вставки текста в окно переводчика
def insert_text(text):
    global driver2
    try:
        # Найти поле для ввода текста по XPath
        input_element = driver2.find_element(By.XPATH, "//div[@id='fakeArea']")

        # Вставить текст из первой страницы в поле на второй странице
        input_element.clear()
        input_element.send_keys(text)
    except Exception as e:
        print("Ошибка при поиске элемента:", e)

# Инициализировать переменную для хранения предыдущего текста
previous_text = ''

# Открыть окно переводчика один раз перед запуском потоков
driver2 = webdriver.Chrome()
url2 = 'https://translate.yandex.ru/?source_lang=en&target_lang=ru'
driver2.get(url2)

# Создать два потока для мониторинга и вставки текста
monitor_thread = threading.Thread(target=monitor_text)
insert_thread = threading.Thread(target=insert_text, args=(previous_text,))  # Передать аргумент

# Запустить оба потока
monitor_thread.start()
insert_thread.start()

# Дождаться завершения обоих потоков
monitor_thread.join()
insert_thread.join()

# Закрыть окно переводчика после завершения потоков
driver2.quit()
