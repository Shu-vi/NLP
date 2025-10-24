"""
Этап 2.
Задача: Реализовать модуль классической векторизации с поддержкой n-грамм.

Указания к выполнению:
Создайте модуль classical_vectorizers.py, выполняющий:

One-Hot Encoding для слов и n-грамм
Bag of Words с различными схемами взвешивания
TF-IDF с настройкой параметров (smooth_idf, sublinear_tf)
Поддержку n-грамм (1–3) и их комбинаций
Анализ разреженности и размерности получаемых матриц
🛠️ Рекомендация: используйте библиотеку scikit-learn (CountVectorizer, TfidfVectorizer).
"""

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import pandas as pd
import json

#Считываем корпус в список для удобства
corpus_filename = "processed_corpus.jsonl"
corpus = []
with open(corpus_filename, "r", encoding="utf-8") as infile:
    for line in infile:
        data = json.loads(line)
        if "text" in data:
            corpus.append(data["text"])

#One-Hot Encoding для слов и n-грамм
one_hot_vectorizer = CountVectorizer(
    ngram_range=(1, 3),#Поддержка 1-3-грамм
    binary=True
)
one_hot_matrix = one_hot_vectorizer.fit_transform(corpus)

print("=== One-Hot Encoding ===")
print("Размерность:", one_hot_matrix.shape)
print("Разреженность:", 1.0 - (one_hot_matrix.nnz / np.prod(one_hot_matrix.shape)))
print(pd.DataFrame(one_hot_matrix.toarray(), columns=one_hot_vectorizer.get_feature_names_out()))

#Bag of Words
bow_vectorizer = CountVectorizer(ngram_range=(1, 3))
bow_matrix = bow_vectorizer.fit_transform(corpus)

print("\n=== Bag of Words (частоты) ===")
print("Размерность:", bow_matrix.shape)
print("Разреженность:", 1.0 - (bow_matrix.nnz / np.prod(bow_matrix.shape)))
print(pd.DataFrame(bow_matrix.toarray(), columns=bow_vectorizer.get_feature_names_out()))

#TF-IDF
tfidf_vectorizer = TfidfVectorizer(
    ngram_range=(1, 3),
    smooth_idf=True,#сглаживание для IDF
    sublinear_tf=True#логарифмическое масштабирование TF
)
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

print("\nTF-IDF")
print("Размерность:", tfidf_matrix.shape)
print("Разреженность:", 1.0 - (tfidf_matrix.nnz / np.prod(tfidf_matrix.shape)))
print(pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out()).round(3))

#Анализ признаков
def analyze_vectorizer(vectorizer, matrix):
    features = vectorizer.get_feature_names_out()
    nonzero = np.count_nonzero(matrix.toarray(), axis=0)
    df = pd.DataFrame({
        "n-грамм": features,
        "nonzero_docs": nonzero,#количество текстов, где слово встречается хотя бы раз
        "idf" : getattr(vectorizer, "idf_", np.nan)
    })
    return df.sort_values(by="nonzero_docs", ascending=False)

print("\nАнализ признаков TF-IDF")
print(analyze_vectorizer(tfidf_vectorizer, tfidf_matrix).head(15))