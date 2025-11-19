from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

# #======================================================
# #====классические алгоритмы. бинарная классификация====
df = pd.read_json("news_ru_binary.jsonl", lines=True)

texts_binary = df["text"].tolist()
labels_binary = df["sentiment"].tolist()
# X_train, X_test, y_train, y_test = train_test_split(texts_binary, labels_binary, test_size=0.2)
#
# model = Pipeline([
#     ('tfidf', TfidfVectorizer(max_features=50000, ngram_range=(1,2))),
#     ('clf', LogisticRegression(max_iter=500))
# ])
#
# model.fit(X_train, y_train)
# pred = model.predict(X_test)
# print(classification_report(y_test, pred))
# #===Мультиклассовая классификация===
df = pd.read_json("news_ru_category.jsonl", lines=True)

texts_category = df["text"].tolist()
labels_category = df["category"].tolist()
# X_train, X_test, y_train, y_test = train_test_split(texts_category, labels_category, test_size=0.2)
#
# model = Pipeline([
#     ('tfidf', TfidfVectorizer(max_features=50000, ngram_range=(1,2))),
#     ('clf', LinearSVC())
# ])
#
# model.fit(X_train, y_train)
# pred = model.predict(X_test)
# print(classification_report(y_test, pred))
#
#
# #===мультилейбл классификация===
df = pd.read_json("news_ru_categorys.jsonl", lines=True)

texts_categorys = df["text"].tolist()
labels_categorys = df["category"].tolist()
#
# mlb = MultiLabelBinarizer()
# labels_categorys = mlb.fit_transform(labels_categorys)
#
# X_train, X_test, y_train, y_test = train_test_split(texts_categorys, labels_categorys, test_size=0.2)
#
# model = Pipeline([
#     ('tfidf', TfidfVectorizer(max_features=60000, ngram_range=(1,2))),
#     ('clf', OneVsRestClassifier(LinearSVC()))
# ])
#
# model.fit(X_train, y_train)
# pred = model.predict(X_test)
# print(classification_report(y_test, pred, target_names=mlb.classes_))

#===================================================
#=====================нейронки=====================
#==========================================binary===
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras import layers, models
import numpy as np

max_tokens = 50000
max_len = 300

vectorizer = TextVectorization(
    max_tokens=max_tokens,
    output_mode="int",
    output_sequence_length=max_len
)

# texts — список строк
vectorizer.adapt(texts_binary)

X = vectorizer(texts_binary)
# y — бинарные метки 0/1
y = np.array([0 if label == "negative" else 1 for label in labels_binary])

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

# model.fit(X, y, epochs=30, batch_size=32, validation_split=0.2)
#========category=====================================
vectorizer = TextVectorization(
    max_tokens=max_tokens,
    output_mode="int",
    output_sequence_length=max_len
)

# texts — список строк
vectorizer.adapt(texts_category)

X = vectorizer(texts_category)

labels_category = [
    0 if label == "политика"
    else 1 if label == "экономика"
    else 2 if label == "спорт"
    else 3 if label == "культура"
    else -1
    for label in labels_category
]
num_classes = len(set(labels_category))      # после LabelEncoder
y = tf.keras.utils.to_categorical(labels_category, num_classes)

model = models.Sequential([
    layers.Embedding(max_tokens, 64),
    layers.Conv1D(64, 5, activation="relu"),
    layers.GlobalMaxPooling1D(),
    layers.Dense(64, activation="relu"),
    layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# model.fit(X, y, epochs=300, batch_size=32, validation_split=0.2)


#=====categorys========================================
vectorizer = TextVectorization(
    max_tokens=max_tokens,
    output_mode="int",
    output_sequence_length=max_len
)

# texts — список строк
vectorizer.adapt(texts_categorys)

X = vectorizer(texts_categorys)

mlb = MultiLabelBinarizer()
labels_categorys = mlb.fit_transform(labels_categorys)
# Y — матрица (samples, num_tags), 0/1
num_tags = labels_categorys.shape[1]

inputs = layers.Input(shape=(max_len,))
x = layers.Embedding(max_tokens, 128)(inputs)
x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dense(128, activation="relu")(x)
outputs = layers.Dense(num_tags, activation="sigmoid")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[tf.keras.metrics.AUC(curve="PR"), "accuracy"]
)

# model.fit(X, labels_categorys, epochs=300, batch_size=32, validation_split=0.2)
#===================================================
#=====================трансформеры=================
#=================binary============================
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

df = pd.read_json("news_ru_binary.jsonl", lines=True)

dataset = Dataset.from_pandas(df)

label2id = {"negative": 0, "positive": 1}
id2label = {0: "negative", 1: "positive"}

def encode_binary(batch):
    batch["label"] = label2id[batch["sentiment"]]
    return batch

dataset_bin = dataset.map(encode_binary)

train_test = dataset_bin.train_test_split(test_size=0.2)
train = train_test["train"]
test = train_test["test"]

model_name = "cointegrated/rubert-tiny"  # очень быстрый

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

trainer.train()

#======category==================================
df = pd.read_json("news_ru_category.jsonl", lines=True)

dataset = Dataset.from_pandas(df)

unique = sorted(set(df["category"]))
label2id = {label: i for i, label in enumerate(unique)}
id2label = {i: label for label, i in label2id.items()}

def encode_multi(batch):
    batch["label"] = label2id[batch["category"]]
    return batch

dataset_mc = dataset.map(encode_multi)
train_test = dataset_mc.train_test_split(test_size=0.2)
train = train_test["train"]
test = train_test["test"]

train_tok = train.map(tokenize, batched=True)
test_tok = test.map(tokenize, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(unique),
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
df = pd.read_json("news_ru_categorys.jsonl", lines=True)

mlb = MultiLabelBinarizer()
binarized = mlb.fit_transform(df["category"])

df["labels"] = binarized.astype(np.float32).tolist()

dataset_ml = Dataset.from_pandas(df)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=256
    )

dataset_ml = dataset_ml.map(tokenize, batched=True)
dataset_ml = dataset_ml.train_test_split(test_size=0.2)
train = dataset_ml["train"]
test = dataset_ml["test"]

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
    num_train_epochs=100
)

# trainer = Trainer(
#     model=model,
#     args=args,
#     train_dataset=train,
#     eval_dataset=test
# )
#
# trainer.train()

