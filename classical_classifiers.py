from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras import layers, models
import numpy as np
from datasets import Dataset
from joblib import dump
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

#======считывание всех датасетов====================
df = pd.read_json("news_ru_binary.jsonl", lines=True)
texts_binary = df["text"].tolist()
labels_binary = df["sentiment"].tolist()
labels_binary = np.array([0 if label == "negative" else 1 for label in labels_binary])
#
df = pd.read_json("news_ru_category.jsonl", lines=True)
texts_category = df["text"].tolist()
labels_category = df["category"].tolist()
labels_category = [
    0 if label == "политика"
    else 1 if label == "экономика"
    else 2 if label == "спорт"
    else 3 if label == "культура"
    else -1
    for label in labels_category
]
num_classes_category = len(set(labels_category))
#
df = pd.read_json("news_ru_categorys.jsonl", lines=True)
texts_categorys = df["text"].tolist()
labels_categorys = df["category"].tolist()
mlb = MultiLabelBinarizer()
labels_categorys = mlb.fit_transform(labels_categorys)
num_tags_categorys = labels_categorys.shape[1]
# labels_categorys = labels_categorys.astype(np.float32).tolist()


#======================================================
#====классические алгоритмы. бинарная классификация====
X_train, X_test, y_train, y_test = train_test_split(texts_binary, labels_binary, test_size=0.2)

model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=50000, ngram_range=(1,2))),
    ('clf', LogisticRegression(max_iter=500))
])

# model.fit(X_train, y_train)
# dump(model, "ml_binary.joblib")
# pred = model.predict(X_test)
# print(classification_report(y_test, pred))

#===Мультиклассовая классификация===
X_train, X_test, y_train, y_test = train_test_split(texts_category, labels_category, test_size=0.2)

model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=50000, ngram_range=(1,2))),
    ('clf', LinearSVC())
])

# model.fit(X_train, y_train)
# dump(model, "ml_category.joblib")
# pred = model.predict(X_test)
# print(classification_report(y_test, pred))

#===мультилейбл классификация===
X_train, X_test, y_train, y_test = train_test_split(texts_categorys, labels_categorys, test_size=0.2)

model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=60000, ngram_range=(1,2))),
    ('clf', OneVsRestClassifier(LinearSVC()))
])

# model.fit(X_train, y_train)
# dump(model, "ml_categorys.joblib")
# pred = model.predict(X_test)
# print(classification_report(y_test, pred, target_names=mlb.classes_))

#===================================================
#=====================нейронки=====================
#==========================================binary===
max_tokens = 50000
max_len = 300

vectorizer = TextVectorization(
    max_tokens=max_tokens,
    output_mode="int",
    output_sequence_length=max_len
)

vectorizer.adapt(texts_binary)
tf.keras.models.save_model(vectorizer, "nn_vectorizer_binary.keras")
X = vectorizer(texts_binary)

model = models.Sequential([
    layers.Embedding(max_tokens, 128),
    layers.Bidirectional(layers.LSTM(64)),
    layers.Dense(64, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC()]
)

model.fit(X, labels_binary, epochs=60, batch_size=32, validation_split=0.2)
model.save("nn_binary.keras")

#========category=====================================
vectorizer = TextVectorization(
    max_tokens=max_tokens,
    output_mode="int",
    output_sequence_length=max_len
)

vectorizer.adapt(texts_category)
tf.keras.models.save_model(vectorizer, "nn_vectorizer_category.keras")
X = vectorizer(texts_category)

y = tf.keras.utils.to_categorical(labels_category, num_classes_category)

model = models.Sequential([
    layers.Embedding(max_tokens, 64),
    layers.Conv1D(64, 5, activation="relu"),
    layers.GlobalMaxPooling1D(),
    layers.Dense(64, activation="relu"),
    layers.Dense(num_classes_category, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
model.fit(X, y, epochs=60, batch_size=32, validation_split=0.2)
model.save("nn_category.keras")

#=====categorys========================================
vectorizer = TextVectorization(
    max_tokens=max_tokens,
    output_mode="int",
    output_sequence_length=max_len
)

vectorizer.adapt(texts_categorys)
tf.keras.models.save_model(vectorizer, "nn_vectorizer_categorys.keras")
X = vectorizer(texts_categorys)

inputs = layers.Input(shape=(max_len,))
x = layers.Embedding(max_tokens, 128)(inputs)
x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dense(128, activation="relu")(x)
outputs = layers.Dense(num_tags_categorys, activation="sigmoid")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[tf.keras.metrics.AUC(curve="PR"), "accuracy"]
)
model.fit(X, labels_categorys, epochs=60, batch_size=32, validation_split=0.2)
model.save("nn_categorys.keras")

#===================================================
#=====================трансформеры=================
#=================binary============================
dataset_bin = Dataset.from_dict({
    "text": texts_binary,
    "labels": labels_binary
})

label2id = {"negative": 0, "positive": 1}
id2label = {0: "negative", 1: "positive"}

train_test = dataset_bin.train_test_split(test_size=0.2, shuffle=True, seed=42)
train = train_test["train"]
test = train_test["test"]

model_name = "cointegrated/rubert-tiny"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=256
    )

train_tok = train.map(tokenize, batched=True)
test_tok = test.map(tokenize, batched=True)
train_tok.set_format("torch")
test_tok.set_format("torch")

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    id2label=id2label,
    label2id=label2id
)
args = TrainingArguments(
    output_dir="./binary_model",
    logging_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=100
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_tok,
    eval_dataset=test_tok
)
# trainer.train()

#======category==================================
dataset = Dataset.from_dict({
    "text": texts_category,
    "labels": labels_category
})

unique_labels = sorted(set(labels_category))
id2label = {i: str(i) for i in unique_labels}
label2id = {str(i): i for i in unique_labels}

train_test = dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)
train = train_test["train"]
test = train_test["test"]

train_tok = train.map(tokenize, batched=True)
test_tok = test.map(tokenize, batched=True)
train_tok.set_format("torch")
test_tok.set_format("torch")

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_classes_category,
    id2label=id2label,
    label2id=label2id
)
args = TrainingArguments(
    output_dir="./category_model",
    logging_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=100
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_tok,
    eval_dataset=test_tok
)
# trainer.train()
#
# preds = trainer.predict(test_tok)
# logits = preds.predictions
# labels = preds.label_ids
#
# y_pred = np.argmax(logits, axis=1)
# accuracy = accuracy_score(labels, y_pred)
#
# print("Accuracy:", accuracy)

#=======================categorys=============
dataset = Dataset.from_dict({
    "text": texts_categorys,
    "labels": labels_categorys
})

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.train_test_split(test_size=0.2)
train = dataset["train"]
test = dataset["test"]
train.set_format("torch")
test.set_format("torch")


model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(mlb.classes_),
    problem_type="multi_label_classification"
)

args = TrainingArguments(
    output_dir="./multilabel_model",
    save_strategy="epoch",
    learning_rate=3e-3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train,
    eval_dataset=test
)
# trainer.train()