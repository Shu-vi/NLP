import tensorflow as tf
import numpy as np
from use_ml import preprocess_text

def predict_sentiment():
    model = tf.keras.models.load_model("nn_binary.keras")
    vectorizer = tf.keras.models.load_model("nn_vectorizer_binary.keras")
    def _inner(text: str) -> str:
        p_text = preprocess_text(text)
        vec = vectorizer([p_text])
        pred = model.predict(vec)[0][0]
        return "positive" if pred >= 0.5 else "negative"
    return _inner

def predict_category():
    model = tf.keras.models.load_model("nn_category.keras")
    vectorizer = tf.keras.models.load_model("nn_vectorizer_category.keras")
    def _inner(text: str) -> str:
        p_text = preprocess_text(text)
        vec = vectorizer([p_text])
        pred = model.predict(vec)[0]
        res = -1
        pred = np.argmax(pred)
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
    model = tf.keras.models.load_model("nn_categorys.keras")
    vectorizer = tf.keras.models.load_model("nn_vectorizer_categorys.keras")
    def _inner(text: str):
        p_text = preprocess_text(text)
        vec = vectorizer([p_text])
        pred = model.predict(vec)[0]
        for i in range(len(pred)):
            if pred[i] >= 0.5:
                pred[i] = 1
            else:
                pred[i] = 0
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