import tensorflow as tf
from use_ml import preprocess_text

def predict_sentiment():
    model = tf.keras.models.load_model("nn_binary.keras")
    vectorizer = tf.keras.models.load_model("nn_vectorizer_binary.keras")
    def _inner(text: str) -> str:
        p_text = preprocess_text(text)
        vec = vectorizer([p_text])
        pred = model.predict(vec)[0][0]
        res = {
            "labels": "positive" if pred >= 0.5 else "negative",
            "probs": pred
        }
        return res
    return _inner

def predict_category():
    model = tf.keras.models.load_model("nn_category.keras")
    vectorizer = tf.keras.models.load_model("nn_vectorizer_category.keras")
    def _inner(text: str) -> str:
        p_text = preprocess_text(text)
        vec = vectorizer([p_text])
        pred = model.predict(vec)[0]
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

def predict_categorys():
    model = tf.keras.models.load_model("nn_categorys.keras")
    vectorizer = tf.keras.models.load_model("nn_vectorizer_categorys.keras")
    def _inner(text: str):
        p_text = preprocess_text(text)
        vec = vectorizer([p_text])
        labels = [
            "политика",
            "экономика",
            "спорт",
            "культура"
        ]
        pred = model.predict(vec)[0]
        res = {
            "labels": labels,
            "probs": pred
        }
        return res
    return _inner