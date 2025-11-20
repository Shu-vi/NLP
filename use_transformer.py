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
        pred = clf(text)
        return pred[0]["label"]
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
        pred = clf(text)
        return pred[0]["label"]

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
        probs = torch.sigmoid(logits).squeeze()
        pred_ids = (probs > 0.5).nonzero().flatten().tolist()
        return [classes[i] for i in pred_ids]

    return _inner