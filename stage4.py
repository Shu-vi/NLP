import time, tracemalloc
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import pandas as pd
import json

#Считываем корпус в список для удобства
corpus_filename = "processed_corpus.jsonl"
corpus = []
results = []

with open(corpus_filename, "r", encoding="utf-8") as infile:
    for line in infile:
        data = json.loads(line)
        if "text" in data:
            corpus.append(data["text"])

def measure_time(func, *args, **kwargs):
    tracemalloc.start()
    t0 = time.perf_counter()
    res = func(*args, **kwargs)
    t1 = time.perf_counter()
    return res, t1 - t0

def measure_mem(dim1, dim2):
    return dim1 * dim2 * 4 / (2**20)#мегабайт

for i in range(1, 4):
    one_hot_vectorizer = CountVectorizer(
        ngram_range=(1, i),
        binary=True
    )
    one_hot_matrix, one_hot_time = measure_time(one_hot_vectorizer.fit_transform, corpus)
    results.append({"метод": f"One-Hot encoding 1-{i}-грамм",
                    "размерность": f"{one_hot_matrix.shape}",
                    "Память": f"{measure_mem(one_hot_matrix.shape[0], one_hot_matrix.shape[1])} мб",
                    "Время": f"{one_hot_time} мс",
                    "Разреженность": f"{one_hot_matrix.nnz / np.prod(one_hot_matrix.shape)}"
                    })

for i in range(1, 4):
    bow_vectorizer = CountVectorizer(
        ngram_range=(1, i)
    )
    bow_matrix, bow_time = measure_time(bow_vectorizer.fit_transform, corpus)
    results.append({"метод": f"Bag of words 1-{i}-грамм",
                    "размерность": f"{bow_matrix.shape}",
                    "Память": f"{measure_mem(bow_matrix.shape[0], bow_matrix.shape[1])} мб",
                    "Время": f"{bow_time} мс",
                    "Разреженность": f"{bow_matrix.nnz / np.prod(bow_matrix.shape)}"
                    })

for i in range(1, 4):
    tfidf_vectorizer = TfidfVectorizer(
        ngram_range=(1, i),
        smooth_idf=True,#сглаживание для IDF
        sublinear_tf=True#логарифмическое масштабирование TF
    )
    tfidf_matrix, tfidf_time = measure_time(tfidf_vectorizer.fit_transform, corpus)
    results.append({"метод": f"tfidf 1-{i}-грамм",
                    "размерность": f"{tfidf_matrix.shape}",
                    "Память": f"{measure_mem(tfidf_matrix.shape[0], tfidf_matrix.shape[1])} мб",
                    "Время": f"{tfidf_time} мс",
                    "Разреженность": f"{tfidf_matrix.nnz / np.prod(tfidf_matrix.shape)}"
                    })

df = pd.DataFrame(results)
df.to_csv("vectorization_metrics.csv", index=False, encoding="utf-8")