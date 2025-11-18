from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

#======================================================
#====классические алгоритмы. бинарная классификация====
df = pd.read_json("news_ru_binary.jsonl", lines=True)

texts_binary = df["text"].tolist()
labels_binary = df["sentiment"].tolist()
X_train, X_test, y_train, y_test = train_test_split(texts_binary, labels_binary, test_size=0.2)

model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=50000, ngram_range=(1,2))),
    ('clf', LogisticRegression(max_iter=500))
])

model.fit(X_train, y_train)
pred = model.predict(X_test)
print(classification_report(y_test, pred))
#===Мультиклассовая классификация===
df = pd.read_json("news_ru_category.jsonl", lines=True)

texts_category = df["text"].tolist()
labels_category = df["category"].tolist()
X_train, X_test, y_train, y_test = train_test_split(texts_category, labels_category, test_size=0.2)

model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=50000, ngram_range=(1,2))),
    ('clf', LinearSVC())
])

model.fit(X_train, y_train)
pred = model.predict(X_test)
print(classification_report(y_test, pred))


#===мультилейбл классификация===
df = pd.read_json("news_ru_categorys.jsonl", lines=True)

texts_categorys = df["text"].tolist()
labels_categorys = df["category"].tolist()

mlb = MultiLabelBinarizer()
labels_categorys = mlb.fit_transform(labels_categorys)

X_train, X_test, y_train, y_test = train_test_split(texts_categorys, labels_categorys, test_size=0.2)

model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=60000, ngram_range=(1,2))),
    ('clf', OneVsRestClassifier(LinearSVC()))
])

model.fit(X_train, y_train)
pred = model.predict(X_test)
print(classification_report(y_test, pred, target_names=mlb.classes_))

#===================================================
#=====================нейронки=====================


#===================================================
#=====================трансформеры=================