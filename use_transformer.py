from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch


def predict_sentiment():
    model_path = "./binary_model/checkpoint-400"
    model_name = "cointegrated/rubert-tiny"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    clf = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        return_all_scores=False
    )
    def _inner(text: str):
        words = text.split()
        truncated_text = ""
        if len(words) > 400:
            truncated_text = ' '.join(words[:400])
        pred = clf(truncated_text)
        res = {
            "labels": pred[0]["label"],
            "probs": pred[0]["score"]
        }
        return res
    return _inner

def predict_category():
    model_path = "./category_model/checkpoint-400"
    model_name = "cointegrated/rubert-tiny"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    clf = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        return_all_scores=False
    )
    def _inner(text: str):
        words = text.split()
        truncated_text = ""
        if len(words) > 400:
            truncated_text = ' '.join(words[:400])
        pred = clf(truncated_text)
        labels = {"политика": 0, "экономика": 0, "спорт": 0, "культура": 0, pred[0]["label"]: pred[0]["score"]}
        classes = [
            "политика",
            "экономика",
            "спорт",
            "культура"
        ]
        new_labels = []
        for cl in classes:
            new_labels.append(labels[cl])
        res = {
            "labels": classes,
            "probs": new_labels
        }
        return res

    return _inner

def predict_categorys():
    model_path = "./multilabel_model/checkpoint-700"
    model_name = "cointegrated/rubert-tiny"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    classes = [
        "политика",
        "экономика",
        "спорт",
        "культура"
    ]
    def _inner(text: str):
        input = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=256
        )
        with torch.no_grad():
            logits = model(**input).logits
        probs = torch.sigmoid(logits).squeeze().tolist()
        res = {
            "labels": classes,
            "probs": probs
        }
        return res


    return _inner