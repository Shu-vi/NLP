#1 привести полученный текст к приемлемому виду
#2 подать текст на вход к модели и получить результат
import spacy
from joblib import load

def predict_sentiment():
    model_binary = load("ml_binary.joblib")
    def _inner(text: str) -> str:
        pred = model_binary.predict([preprocess_text(text)])[0]
        return "positive" if pred == 1 else "negative"
    return _inner



def predict_category():
    model_category = load("ml_category.joblib")

    def _inner(text: str) -> str:
        pred = model_category.predict([preprocess_text(text)])[0]
        res = -1
        if pred == 0:
            res = "политика"
        elif pred == 1:
            res = "экономика"
        elif pred == 2:
            res = "спорт"
        elif pred == 3:
            res = "культура"
        return res
    return _inner

def predict_categorys():
    model_categorys = load("ml_categorys.joblib")

    def _inner(text: str):
        pred = model_categorys.predict([preprocess_text(text)])[0]
        res = []
        if pred[0] == 1:
            res.append("политика")
        elif pred[1] == 1:
            res.append("экономика")
        elif pred[2] == 1:
            res.append("спорт")
        elif pred[3] == 1:
            res.append("культура")
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