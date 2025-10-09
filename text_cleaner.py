"""
3.2. Этап 2. Предварительная обработка и очистка текста
Задача: Реализовать модуль первичной очистки текста от нетекстовых элементов и его нормализации.

Указания к выполнению:
Создайте модуль text_cleaner.py, выполняющий:

Удаление HTML-разметки, служебных символов, рекламных блоков
Стандартизацию пробельных символов
Приведение текста к нижнему регистру (опционально — зависит от метода токенизации)
Фильтрацию стоп-слов (например, с использованием nltk.corpus.stopwords)
"""

import re
import nltk
from nltk.corpus import stopwords

# Убедимся, что стоп-слова загружены
try:
    _ = stopwords.words("russian")
except LookupError:
    nltk.download("stopwords")

"""
Очищает и нормализует текст статьи
- Удаляет HTML теги, спецсимволы, рекламные фразы
- Приводит пробелы к стандарту
- Приводит к нижнему регистру (опционально)
- Удаляет стоп-слова (опционально)
"""
def clean_text(text: str, lowercase: bool = True, remove_stopwords: bool = True) -> str:
    #удаляем HTML-теги
    text = re.sub(r"<[^>]+>", " ", text)

    #удаляем служебные символы, ссылки и спецзнаки
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-zA-Zа-яА-Я0-9\s.,!?]", " ", text)

    #убираем множественные пробелы
    text = re.sub(r"\s+", " ", text).strip()

    # Приводим к нижнему регистру, если нужно
    if lowercase:
        text = text.lower()

    # Удаляем стоп-слова (если нужно)
    if remove_stopwords:
        stop_words = set(stopwords.words("russian"))
        text = " ".join(word for word in text.split() if word not in stop_words)

    return text
