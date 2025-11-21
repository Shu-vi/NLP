#1 привести полученный текст к приемлемому виду
#2 подать текст на вход к модели и получить результат
import spacy
from joblib import load

def predict_sentiment():
    model_binary = load("ml_binary.joblib")
    def _inner(text: str):
        pred = model_binary.predict([preprocess_text(text)])[0]
        res = {
            "labels": "positive" if pred == 1 else "negative",
            "probs": pred
        }
        return res
    return _inner



def predict_category():
    model_category = load("ml_category.joblib")

    def _inner(text: str):
        pred = model_category.predict([preprocess_text(text)])[0]
        labels = [
            "политика",
            "экономика",
            "спорт",
            "культура"
        ]
        probs = [0, 0, 0, 0]
        probs[pred] = 1
        res = {
            "labels": labels,
            "probs": probs
        }
        return res
    return _inner

def predict_categorys():
    model_categorys = load("ml_categorys.joblib")

    def _inner(text: str):
        pred = model_categorys.predict([preprocess_text(text)])[0]
        labels = [
            "политика",
            "экономика",
            "спорт",
            "культура"
        ]
        res = {
            "labels": labels,
            "probs": pred
        }
        return res
    return _inner

def preprocess_text(text: str) -> str:
    if text is None:
        return ""

    nlp = spacy.load("ru_core_news_md", disable=["ner"])

    text = " ".join(text.split()).lower()

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