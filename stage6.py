from gensim.models import Word2Vec, FastText, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.utils import simple_preprocess
from sklearn.metrics.pairwise import cosine_similarity
import json
import numpy as np
import pandas as pd

SKIPGRAM = "skipgram"
CBOW = "sbow"
PVDM = "pv-dm"
PVDBOW = "pv-dbow"
#корпус
corpus_filename = "processed_corpus.jsonl"
corpus = []
with open(corpus_filename, "r", encoding="utf-8") as infile:
    for line in infile:
        data = json.loads(line)
        if "text" in data:
            corpus.append(data["text"])

#токенизация для word-level
sentences = [simple_preprocess(doc) for doc in corpus]

#подготовка для document-level. Каждому доку нужен уникльный тег
tagged_docs = [
    TaggedDocument(words=simple_preprocess(doc), tags=[f"DOC_{i}"])
    for i, doc in enumerate(corpus)
]


def get_word_2_vec(sentences, vector_size, window, min_count, model):
    sg = 0
    if model == SKIPGRAM:
        sg = 1
    elif model == CBOW:
        sg = 0
    return Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg
    )

def get_fast_text(sentences, vector_size, window, min_count, model):
    sg = 0
    if model == SKIPGRAM:
        sg = 1
    elif model == CBOW:
        sg = 0
    return FastText(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg
    )

def get_doc_2_vec(documents, vector_size, window, min_count, model, epochs):
    dm = 0
    if model == PVDM:
        dm = 1
    elif model == PVDBOW:
        dm = 0
    return Doc2Vec(
        documents=documents,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        dm=dm,
        epochs=epochs
    )

def sim_doc2vec(model, sent1, sent2):
    v1 = model.infer_vector(simple_preprocess(sent1))
    v2 = model.infer_vector(simple_preprocess(sent2))
    return cosine_similarity([v1], [v2])[0][0]

def similarity_matrix(model, words):
    vectors = np.array([model.wv[w] for w in words])
    return cosine_similarity(vectors)

def sim_axios(str1, str2, model, words):
    axis = model_w2v.wv[str1] - model_w2v.wv[str2]
    scores = {}
    for w in words:
        if w in model.wv:
            vec = model.wv[w]
            projection = cosine_similarity([vec], [axis])[0][0]
            scores[w] = projection
    return scores

model_w2v = get_word_2_vec(sentences, 100, 5, 5, SKIPGRAM)
model_fasttext = get_fast_text(sentences, 100, 5, 5, SKIPGRAM)
model_doc2vec = get_doc_2_vec(tagged_docs, 100, 5, 5, PVDM, 20)
model_w2v.save("./models/word2vec.model")
model_fasttext.save("./models/fasttext.model")
model_doc2vec.save("./models/doc2vec.model")
#семантическое сходство
print("Сходство 'путин'–'президент':", model_w2v.wv.similarity("путин", "президент"))
print("Сходство 'путин'–'президент':", model_fasttext.wv.similarity("путин", "президент"))
print("Сходство: 'президент российский федерации поел утром'-'путин позавтракал'", sim_doc2vec(model_doc2vec, "президент российский федерации поел утром", "путин позавтракал"))
#интерактивная векторная арифметика
#Матрица семантической близости
words = ["россия", "трамп", "китай", "спорт"]

df = pd.DataFrame(similarity_matrix(model_w2v, words), index=words, columns=words)
print("\nМатрица семантической близости w2v:")
print(df.round(2))

df = pd.DataFrame(similarity_matrix(model_fasttext, words), index=words, columns=words)
print("\nМатрица семантической близости fasttext:")
print(df.round(2))

print("россия - москва + вашингтон = ?", model_w2v.wv.most_similar(positive=["вашингтон", "россия"], negative=["москва"], topn=3))
print("россия - москва + вашингтон = ?", model_fasttext.wv.most_similar(positive=["вашингтон", "россия"], negative=["москва"], topn=3))

#визуализация симантических осей
words = ["tomahawk", "украина", "медицина", "бесплатный", "наука"]
df = pd.DataFrame.from_dict(sim_axios("европа", "россия", model_w2v, words), orient='index', columns=['projection']).sort_values('projection', ascending=False)
print(df)

df = pd.DataFrame.from_dict(sim_axios("европа", "россия", model_fasttext, words), orient='index', columns=['projection']).sort_values('projection', ascending=False)
print(df)