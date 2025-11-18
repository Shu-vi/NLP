import os
import json
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm

import newspaper
from newspaper import Article, build
from bs4 import BeautifulSoup

import spacy
from spacy.language import Language

import numpy as np

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

from sentence_transformers import SentenceTransformer

# -----------------------
# Конфигурация
# -----------------------
SEED_SITES = [
    "https://lenta.ru",
    "https://ria.ru",
    "https://tass.ru",
    "https://www.kommersant.ru",
    "https://www.sport-express.ru",
    "https://www.mk.ru",
]
OUTPUT_JSONL = "news_ru_category.jsonl"
MIN_WORDS_PER_ARTICLE = 150
TARGET_TOTAL_WORDS = 50_000
MAX_ARTICLES_PER_SITE = 500

# zero-shot модель (мультилингвальная)
ZS_MODEL = "joeddav/xlm-roberta-large-xnli"
LABELS = ["политика", "экономика", "спорт", "культура"]

# порог уверенности для zero-shot (если ниже — применяем эвристику)
LABEL_CONF_THRESHOLD = 0.60

# embedding model (поддерживает русский)
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# -----------------------
# Предобработка текста
# -----------------------
def preprocess_text(raw_html_or_text: str, nlp: Language) -> str:
    """
    Очищает HTML (если есть), приводит к lowercase, токенизирует и лемматизирует,
    удаляет стоп-слова и пунктуацию. Возвращает строку из лемм через пробел.
    """
    if not raw_html_or_text:
        return ""

    # remove HTML if present
    soup = BeautifulSoup(raw_html_or_text, "html.parser")
    text = soup.get_text(separator=" ")
    text = " ".join(text.split())
    text = text.lower()

    doc = nlp(text)

    lemmas = []
    for tok in doc:
        if tok.is_stop or tok.is_punct or tok.is_space:
            continue
        lemma = tok.lemma_.strip()
        if len(lemma) <= 1:
            continue
        # filter numbers-only tokens, urls etc.
        if lemma.isnumeric():
            continue
        lemmas.append(lemma)

    return " ".join(lemmas)


# -----------------------
# Keyword-based fallback для категорий
# -----------------------
CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "политика": ["президент", "парламент", "выборы", "правительство", "депутат", "министерство", "санкции", "мвд", "госдума", "власть"],
    "экономика": ["инфляция", "рынок", "курс", "банк", "экономика", "вложение", "акция", "товар", "поставка", "производство", "инвестиции"],
    "спорт": ["матч", "гол", "чемпионат", "футбол", "баскетбол", "атлет", "олимпиада", "тренер", "счет", "победил"],
    "культура": ["фильм", "театр", "кино", "выставка", "музей", "концерт", "музыка", "литература", "книга", "режиссер", "художник"]
}

def keyword_classify(text: str) -> Optional[str]:
    """
    Простая эвристика: считает упоминания ключевых слов по категориям.
    Возвращает категорию с наибольшим числом совпадений, или None.
    """
    counts = {cat: 0 for cat in CATEGORY_KEYWORDS}
    for cat, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            counts[cat] += text.count(kw)
    # выберем максимум
    best_cat = max(counts, key=lambda k: counts[k])
    if counts[best_cat] == 0:
        return None
    return best_cat


# -----------------------
# Сбор ссылок и статьи
# -----------------------
def collect_article_urls(site_url: str, max_articles: int = MAX_ARTICLES_PER_SITE) -> List[str]:
    try:
        paper = build(site_url, memoize_articles=False, language="ru")
    except Exception:
        return []
    urls = []
    for art in paper.articles[:max_articles]:
        if art.url:
            urls.append(art.url)
    # unique
    return list(dict.fromkeys(urls))


def fetch_article(url: str) -> Tuple[Optional[str], Optional[str]]:
    try:
        a = Article(url, language="ru")
        a.download()
        a.parse()
        title = a.title or ""
        text = a.text or ""
        if not text and a.html:
            soup = BeautifulSoup(a.html, "html.parser")
            text = soup.get_text(separator=" ")
        return title, text
    except Exception:
        return None, None


# -----------------------
# Embedding
# -----------------------
class Embedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        texts: list of strings
        returns: numpy.ndarray shape=(len(texts), dim)
        """
        embs = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embs


# -----------------------
# Основной pipeline: сбор, предобработка, классификация, запись
# -----------------------
def build_dataset(seed_sites: List[str],
                  out_path: str = OUTPUT_JSONL,
                  min_words: int = MIN_WORDS_PER_ARTICLE,
                  target_total_words: int = TARGET_TOTAL_WORDS):
    # spaCy russian model
    nlp = spacy.load("ru_core_news_md", disable=["ner"])
    embedder = Embedder()  # можно использовать для получения эмбеддингов при необходимости
    total_words = 0
    written = 0
    seen = set()

    with open(out_path, "w", encoding="utf-8") as fout:
        for site in seed_sites:
            print(f"Сканирую сайт: {site}")
            urls = collect_article_urls(site)
            print(f"Найдено кандидатов: {len(urls)}")

            for url in tqdm(urls):
                if total_words >= target_total_words:
                    break
                if url in seen:
                    continue
                seen.add(url)

                title_raw, raw_text = fetch_article(url)
                if not title_raw or not raw_text:
                    continue

                processed = preprocess_text(raw_text, nlp)
                words = processed.split()
                if len(words) < min_words:
                    continue

                # fallback по ключевым словам на предобработанном тексте
                kw_label = keyword_classify(processed)
                if kw_label is not None:
                    assigned_label = kw_label
                else:
                    continue

                # final record (никаких None)
                record = {
                    "title": title_raw.strip() if title_raw else "",
                    "text": processed.strip() if processed else "",
                    "category": assigned_label
                }

                # sanity checks
                if not record["title"] or not record["text"] or not record["category"]:
                    continue

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
                total_words += len(words)

            if total_words >= target_total_words:
                break

    print(f"Готово. Записано статей: {written}, суммарно слов (примерно): {total_words}")
    print("Выходной файл:", os.path.abspath(out_path))


# -----------------------
# Пример использования
# -----------------------
if __name__ == "__main__":
    build_dataset(SEED_SITES, OUTPUT_JSONL)
    # Пример получения эмбеддингов для первых 10 статей
    def load_jsonl(path: str, limit: int = 10):
        out = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= limit:
                    break
                out.append(json.loads(line))
        return out

    articles = load_jsonl(OUTPUT_JSONL, limit=10)
    if articles:
        texts = [a["text"] for a in articles]
        emb = Embedder()
        vectors = emb.embed(texts)
        print("Embeddings shape:", vectors.shape)
    else:
        print("Файл пуст или отсутствует.")
