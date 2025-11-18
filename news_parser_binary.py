import os
import json
import random
from typing import List, Tuple, Optional
from newspaper import Article, build
from bs4 import BeautifulSoup
import spacy
from spacy.language import Language
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


# -----------------------------------
# Сайты для парсинга новостей
# -----------------------------------
SEED_SITES = [
    "https://ria.ru/",
    "https://tass.ru/",
    "https://www.kommersant.ru/",
    "https://lenta.ru/",
    "https://iz.ru/",
]

OUTPUT_JSONL = "news_ru_binary.jsonl"
MIN_WORDS_PER_ARTICLE = 150
TARGET_TOTAL_WORDS = 50_000
MAX_ARTICLES_PER_SITE = 500

SBERT_MODEL_NAME = "cointegrated/rubert-tiny2"   # быстрый русскоязычный sBERT


# -----------------------------------
# Sentiment model (Russian)
# -----------------------------------
class RussianSentiment:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny-sentiment-balanced")
        self.model = AutoModelForSequenceClassification.from_pretrained("cointegrated/rubert-tiny-sentiment-balanced")

        self.labels = {
            0: "negative",
            1: "neutral",
            2: "positive"
        }

    def get(self, text: str) -> str:
        if not text.strip():
            return "negative"

        inputs = self.tokenizer(text[:3000], padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**inputs).logits
        pred = torch.argmax(logits, axis=1).item()

        # Чтобы строго вернуть только положительный/негативный:
        if pred == 2:
            return "positive"
        elif pred == 1:
            return random.choice(["positive", "negative"])
        else:
            return "negative"


# -----------------------------------
# Preprocessing (Russian)
# -----------------------------------
def preprocess_text(raw_html: str, nlp: Language) -> str:
    if raw_html is None:
        return ""

    soup = BeautifulSoup(raw_html, "html.parser")
    text = soup.get_text(separator=" ")

    text = " ".join(text.split())
    text = text.lower()

    doc = nlp(text)

    tokens = []
    for t in doc:
        if t.is_stop or t.is_punct or t.is_space:
            continue
        lemma = t.lemma_.strip()
        if len(lemma) <= 1:
            continue
        tokens.append(lemma)

    return " ".join(tokens)


def collect_articles_from_site(site_url: str, max_articles: int = MAX_ARTICLES_PER_SITE) -> List[str]:
    try:
        paper = build(site_url, memoize_articles=False, language="ru")
    except Exception:
        return []

    urls = []
    for art in paper.articles[:max_articles]:
        if art.url:
            urls.append(art.url)

    return list(dict.fromkeys(urls))


def fetch_article_content(url: str) -> Tuple[Optional[str], Optional[str]]:
    try:
        a = Article(url, language="ru")
        a.download()
        a.parse()
        title = a.title or ""
        text = a.text or ""

        if not text and a.html:
            soup = BeautifulSoup(a.html, "html.parser")
            text = soup.get_text(" ")
        return title, text
    except Exception:
        return None, None


# -----------------------------------
# MAIN PIPELINE
# -----------------------------------
def build_dataset(seed_sites: List[str],
                  out_path: str = OUTPUT_JSONL,
                  min_words_per_article: int = MIN_WORDS_PER_ARTICLE,
                  target_total_words: int = TARGET_TOTAL_WORDS):
    def is_add(label, dict):
        if label == "positive":
            if dict["positive"] / dict["negative"] < 1.25:
                return True
            else:
                return False
        elif label == "negative":
            if dict["negative"] / dict["positive"] < 1.25:
                return True
            else:
                return False
        return False

    nlp = spacy.load("ru_core_news_md", disable=["ner"])
    sentiment = RussianSentiment()
    count_classes = {
        "positive": 1,
        "negative": 1
    }
    total_words = 0
    written = 0
    seen_urls = set()

    with open(out_path, "w", encoding="utf-8") as fout:
        for site in seed_sites:
            print(f"\nСканирую: {site}")
            urls = collect_articles_from_site(site)
            print(f"Найдено ссылок: {len(urls)}")

            for url in tqdm(urls):
                if total_words >= target_total_words:
                    break
                if url in seen_urls:
                    continue
                seen_urls.add(url)

                title_raw, raw_text = fetch_article_content(url)
                if not title_raw or not raw_text:
                    continue

                processed = preprocess_text(raw_text, nlp)
                words = processed.split()
                if len(words) < min_words_per_article:
                    continue

                sentiment_label = sentiment.get(processed)
                if not is_add(sentiment_label, count_classes):
                    continue
                count_classes[sentiment_label] += 1
                record = {
                    "title": title_raw.strip(),
                    "text": processed.strip(),
                    "sentiment": sentiment_label
                }

                if not record["title"] or not record["text"] or not record["sentiment"]:
                    continue

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
                total_words += len(words)

            if total_words >= target_total_words:
                break

    print(f"\nГотово. Статей записано: {written}, суммарное число слов: {total_words}")
    print("Файл:", os.path.abspath(out_path))


# -----------------------------------
# Run
# -----------------------------------
if __name__ == "__main__":
    build_dataset(SEED_SITES)
