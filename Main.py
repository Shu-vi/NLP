import json
from Vectorization import create_tfidf, create_w2v, create_fasttext
from Tokenizer import create_bpe, tokenize_naive, tokenize_regex
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from Clustering import k_means, agglomerative_clustering, spectral_clustering, mini_batch_means, use_hdbscan
import numpy as np

def get_metrics(X_emb, labels):
    """
    Silhouette Score, ближе к 1 — отлично, около 0 — слабое разделение, <0 — кластеры плохие
    Calinski–Harabasz, чем больше, тем лучше
    Davies–Bouldin, чем меньше, тем лучше, идеал → 0
    """
    try:
        sil = silhouette_score(X_emb, labels)
    except:
        sil = None
    try:
        ch = calinski_harabasz_score(X_emb, labels)
    except:
        ch = None
    try:
        db = davies_bouldin_score(X_emb, labels)
    except:
        db = None

    print("Silhouette:", sil)
    print("Calinski-Harabasz:", ch)
    print("Davies-Bouldin:", db)
    return sil, ch, db

bpe = create_bpe()

tfidf = create_tfidf()
w2v  = create_w2v()
fasttext = create_fasttext()

corpus = []
with open("processed_corpus.jsonl", "r", encoding="utf-8") as infile:
    for line in infile:
        data = json.loads(line)
        if "text" in data:
            corpus.append(data["text"])

#токенизация
list_tokens_whitespace = [tokenize_naive(text) for text in corpus]
list_tokens_regex = [tokenize_regex(text) for text in corpus]
list_tokens_bpe = [bpe(text) for text in corpus]

sentences_whitespace = [" ".join(tokens) for tokens in list_tokens_whitespace]
sentences_regex      = [" ".join(tokens) for tokens in list_tokens_regex]
sentences_bpe        = [" ".join(tokens) for tokens in list_tokens_bpe]

#векторизация
#tf-idf
tfidf_whitespace = tfidf(sentences_whitespace)
tfidf_regex = tfidf(sentences_regex)
tfidf_bpe = tfidf(sentences_bpe)
#w2v
w2v_whitespace = []
for tokens in list_tokens_whitespace:
    doc_embed = []
    for token in tokens:
        token_embedding = w2v(token)
        if token_embedding is not None:
            doc_embed.append(token_embedding)
    w2v_whitespace.append(np.mean(doc_embed, axis=0))
w2v_regex = []
for tokens in list_tokens_regex:
    doc_embed = []
    for token in tokens:
        token_embedding = w2v(token)
        if token_embedding is not None:
            doc_embed.append(token_embedding)
    w2v_regex.append(np.mean(doc_embed, axis=0))
w2v_bpe = []
for tokens in list_tokens_bpe:
    doc_embed = []
    for token in tokens:
        token_embedding = w2v(token)
        if token_embedding is not None:
            doc_embed.append(token_embedding)
    w2v_bpe.append(np.mean(doc_embed, axis=0))
#fasttext
fasttext_whitespace = []
for tokens in list_tokens_whitespace:
    doc_embed = []
    for token in tokens:
        token_embedding = fasttext(token)
        if token_embedding is not None:
            doc_embed.append(token_embedding)
    fasttext_whitespace.append(np.mean(doc_embed, axis=0))
fasttext_regex = []
for tokens in list_tokens_whitespace:
    doc_embed = []
    for token in tokens:
        token_embedding = fasttext(token)
        if token_embedding is not None:
            doc_embed.append(token_embedding)
    fasttext_regex.append(np.mean(doc_embed, axis=0))
fasttext_bpe = []
for tokens in list_tokens_whitespace:
    doc_embed = []
    for token in tokens:
        token_embedding = fasttext(token)
        if token_embedding is not None:
            doc_embed.append(token_embedding)
    fasttext_bpe.append(np.mean(doc_embed, axis=0))
#кластеризация
#kmeans
print("kmeans")
get_metrics(tfidf_whitespace, k_means(tfidf_whitespace, 5))
get_metrics(tfidf_regex, k_means(tfidf_regex, 5))
get_metrics(tfidf_bpe, k_means(tfidf_bpe, 5))
print()
get_metrics(w2v_whitespace, k_means(w2v_whitespace, 5))
get_metrics(w2v_regex, k_means(w2v_regex, 5))
get_metrics(w2v_bpe, k_means(w2v_bpe, 5))
print()
get_metrics(fasttext_whitespace, k_means(fasttext_whitespace, 5))
get_metrics(fasttext_regex, k_means(fasttext_regex, 5))
get_metrics(fasttext_bpe, k_means(fasttext_bpe, 5))
#mini batch means
print("mini batch means")
get_metrics(tfidf_whitespace, mini_batch_means(tfidf_whitespace, 5))
get_metrics(tfidf_regex, mini_batch_means(tfidf_regex, 5))
get_metrics(tfidf_bpe, mini_batch_means(tfidf_bpe, 5))
print()
get_metrics(w2v_whitespace, mini_batch_means(w2v_whitespace, 5))
get_metrics(w2v_regex, mini_batch_means(w2v_regex, 5))
get_metrics(w2v_bpe, mini_batch_means(w2v_bpe, 5))
print()
get_metrics(fasttext_whitespace, mini_batch_means(fasttext_whitespace, 5))
get_metrics(fasttext_regex, mini_batch_means(fasttext_regex, 5))
get_metrics(fasttext_bpe, mini_batch_means(fasttext_bpe, 5))
#hdbscan
print("hdbscan")
get_metrics(tfidf_whitespace, use_hdbscan(tfidf_whitespace))
get_metrics(tfidf_regex, use_hdbscan(tfidf_regex))
get_metrics(tfidf_bpe, use_hdbscan(tfidf_bpe))
print()
get_metrics(w2v_whitespace, use_hdbscan(w2v_whitespace))
get_metrics(w2v_regex, use_hdbscan(w2v_regex))
get_metrics(w2v_bpe, use_hdbscan(w2v_bpe))
print()
get_metrics(fasttext_whitespace, use_hdbscan(fasttext_whitespace))
get_metrics(fasttext_regex, use_hdbscan(fasttext_regex))
get_metrics(fasttext_bpe, use_hdbscan(fasttext_bpe))
#agglomerative clustering
print("agglomerative clustering")
get_metrics(tfidf_whitespace, agglomerative_clustering(tfidf_whitespace))
get_metrics(tfidf_regex, agglomerative_clustering(tfidf_regex))
get_metrics(tfidf_bpe, agglomerative_clustering(tfidf_bpe))
print()
get_metrics(w2v_whitespace, agglomerative_clustering(w2v_whitespace))
get_metrics(w2v_regex, agglomerative_clustering(w2v_regex))
get_metrics(w2v_bpe, agglomerative_clustering(w2v_bpe))
print()
get_metrics(fasttext_whitespace, agglomerative_clustering(fasttext_whitespace))
get_metrics(fasttext_regex, agglomerative_clustering(fasttext_regex))
get_metrics(fasttext_bpe, agglomerative_clustering(fasttext_bpe))
#spectral_clustering
print("spectral_clustering")
get_metrics(tfidf_whitespace, spectral_clustering(tfidf_whitespace))
get_metrics(tfidf_regex, spectral_clustering(tfidf_regex))
get_metrics(tfidf_bpe, spectral_clustering(tfidf_bpe))
print()
get_metrics(w2v_whitespace, spectral_clustering(w2v_whitespace))
get_metrics(w2v_regex, spectral_clustering(w2v_regex))
get_metrics(w2v_bpe, spectral_clustering(w2v_bpe))
print()
get_metrics(fasttext_whitespace, spectral_clustering(fasttext_whitespace))
get_metrics(fasttext_regex, spectral_clustering(fasttext_regex))
get_metrics(fasttext_bpe, spectral_clustering(fasttext_bpe))