"""
3.3. Этап 3. Проектирование универсального модуля предобработки
Задача: Разработать конфигурируемый модуль universal_preprocessor.py для приведения текста к единому стандарту перед токенизацией.

Указания к выполнению:
Модуль должен обеспечивать:

Стандартизацию пунктуации и пробелов
Замену числительных, URL-адресов и email на унифицированные токены:
<NUM>, <URL>, <EMAIL>
Обработку общеязыковых и специальных сокращений (например: “т.е.” → “то есть”, “г.” → “год”)
🛠️ Рекомендация: используйте библиотеку re (регулярные выражения) для гибкой настройки правил.
"""

import re
import html
from typing import Dict

class UniversalPreprocessor:
    """
    Универсальный препроцессор текста.
    Приводит текст к единому стандарту перед токенизацией.
    """
    def __init__(self,
                 lowercase: bool = True,
                 normalize_punctuation: bool = True,
                 replace_patterns: bool = True,
                 expand_abbreviations: bool = True):
        self.lowercase = lowercase
        self.normalize_punctuation = normalize_punctuation
        self.replace_patterns = replace_patterns
        self.expand_abbreviations = expand_abbreviations

        # Словарь сокращений (можно расширять)
        self.abbreviations: Dict[str, str] = {
            r"\bт\.е\.": "то есть",
            r"\bт\.к\.": "так как",
            r"\bг\.": "год",
            r"\bдр\.": "другие",
            r"\bт\.д\.": "так далее",
            r"\bт\.п\.": "тому подобное",
            r"\bт\.н\.": "так называемый",
            r"\bт\.ч\.": "том числе",
            r"\bрф\b": "российская федерация",
            r"\bсша\b": "соединенные штаты америки",
            r"\bес\b": "европейский союз",
        }

        # Паттерны замены на унифицированные токены
        self.patterns: Dict[str, str] = {
            r"\b\d+(?:[\.,]\d+)?\b": "<NUM>",                # Числа, включая десятичные
            r"(https?://\S+|www\.\S+)": "<URL>",            # URL-адреса
            r"\b[\w\.-]+@[\w\.-]+\.\w+\b": "<EMAIL>",       # Email-адреса
        }

    def _normalize_punctuation(self, text: str) -> str:
        """Приводит пунктуацию и пробелы к стандартному виду."""
        text = html.unescape(text)  # заменяем HTML-сущности (&nbsp; → пробел)
        text = text.replace("—", "-").replace("–", "-")
        text = re.sub(r"[“”«»]", '"', text)   # кавычки в стандартные
        text = re.sub(r"\s+", " ", text)       # множественные пробелы → один
        text = re.sub(r"\s([,.!?;:])", r"\1", text)  # убираем пробел перед пунктуацией
        text = re.sub(r"([,.!?;:])([^\s])", r"\1 \2", text)  # пробел после пунктуации
        return text.strip()

    def _replace_patterns(self, text: str) -> str:
        """Заменяет числа, URL, email и т.п. на токены."""
        for pattern, token in self.patterns.items():
            text = re.sub(pattern, token, text, flags=re.I)
        return text

    def _expand_abbreviations(self, text: str) -> str:
        """Заменяет общеязыковые и специальные сокращения."""
        for abbr, expanded in self.abbreviations.items():
            text = re.sub(abbr, expanded, text, flags=re.I)
        return text

    def process(self, text: str) -> str:
        """Основной метод препроцессинга."""
        if not text:
            return ""

        # 1. Приведение к нижнему регистру
        if self.lowercase:
            text = text.lower()

        # 2. Нормализация пунктуации и пробелов
        if self.normalize_punctuation:
            text = self._normalize_punctuation(text)

        # 3. Замена шаблонов (числа, URL, email и т.д.)
        if self.replace_patterns:
            text = self._replace_patterns(text)

        # 4. Расширение сокращений
        if self.expand_abbreviations:
            text = self._expand_abbreviations(text)



        return text.strip()


"""
Этап 3 запуск.
"""

import json
from tqdm import tqdm

# Инициализация твоего препроцессора
up = UniversalPreprocessor()

# Пути к файлам
input_path = "corpus.jsonl"       # исходный файл
output_path = "processed.jsonl"  # выходной файл

# Открываем входной и выходной файлы построчно
with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
    for line in tqdm(infile, desc="Processing articles"):
        # Пропускаем пустые строки
        line = line.strip()
        if not line:
            continue

        # Загружаем JSON-объект
        try:
            article = json.loads(line)
        except json.JSONDecodeError:
            print("Пропущена некорректная строка")
            continue

        # Проверяем наличие поля text
        article_text = article.get("text", "")
        if article_text:
            try:
                article["text"] = up.process(article_text)
            except Exception as e:
                print(f"Ошибка обработки текста: {e}")
                continue

        # Сохраняем обратно в jsonl
        json.dump(article, outfile, ensure_ascii=False)
        outfile.write("\n")
