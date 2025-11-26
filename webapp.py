import streamlit as st
import json
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from Vectorization import create_tfidf, create_w2v, create_fasttext
from Tokenizer import create_bpe, tokenize_naive, tokenize_regex
from Clustering import k_means, agglomerative_clustering, spectral_clustering, mini_batch_means, use_hdbscan


# Загрузка моделей один раз при запуске
@st.cache_resource
def load_models():
    bpe = create_bpe()
    tfidf_func, tfidf_vectorizer = create_tfidf()
    w2v_func, w2v_model = create_w2v()
    fasttext_func, fasttext_model = create_fasttext()
    return bpe, tfidf_func, tfidf_vectorizer, w2v_func, w2v_model, fasttext_func, fasttext_model


@st.cache_data
def load_corpus():
    corpus = []
    with open("processed_corpus.jsonl", "r", encoding="utf-8") as infile:
        for line in infile:
            data = json.loads(line)
            if "text" in data:
                corpus.append(data["text"])
    return corpus


def get_metrics(X_emb, labels):
    """Вычисление метрик кластеризации"""
    metrics = {}
    try:
        metrics['silhouette'] = silhouette_score(X_emb, labels)
    except:
        metrics['silhouette'] = None
    try:
        metrics['calinski_harabasz'] = calinski_harabasz_score(X_emb, labels)
    except:
        metrics['calinski_harabasz'] = None
    try:
        metrics['davies_bouldin'] = davies_bouldin_score(X_emb, labels)
    except:
        metrics['davies_bouldin'] = None
    return metrics


def vectorize_text(corpus, tokenization_method, vectorization_method, models):
    bpe, tfidf_func, tfidf_vectorizer, w2v_func, w2v_model, fasttext_func, fasttext_model = models

    # Токенизация
    if tokenization_method == "Naive (whitespace)":
        tokens_list = [tokenize_naive(text) for text in corpus]
        sentences = [" ".join(tokens) for tokens in tokens_list]
    elif tokenization_method == "Regex":
        tokens_list = [tokenize_regex(text) for text in corpus]
        sentences = [" ".join(tokens) for tokens in tokens_list]
    else:  # BPE
        tokens_list = [bpe(text) for text in corpus]
        sentences = [" ".join(tokens) for tokens in tokens_list]

    # Векторизация
    if vectorization_method == "TF-IDF":
        embeddings = tfidf_func(sentences)
        return embeddings, tokens_list, sentences, tfidf_vectorizer

    elif vectorization_method == "Word2Vec":
        embeddings = []
        for tokens in tokens_list:
            doc_embed = []
            for token in tokens:
                token_embedding = w2v_func(token)
                if token_embedding is not None:
                    doc_embed.append(token_embedding)
            if len(doc_embed) > 0:
                embeddings.append(np.mean(doc_embed, axis=0))
            else:
                embeddings.append(np.zeros(300))
        return np.array(embeddings), tokens_list, sentences, w2v_model  # возвращаем модель

    else:  # FastText
        embeddings = []
        for tokens in tokens_list:
            doc_embed = []
            for token in tokens:
                token_embedding = fasttext_func(token)
                if token_embedding is not None:
                    doc_embed.append(token_embedding)
            if len(doc_embed) > 0:
                embeddings.append(np.mean(doc_embed, axis=0))
            else:
                embeddings.append(np.zeros(300))
        return np.array(embeddings), tokens_list, sentences, fasttext_model  # возвращаем модель

def get_top_tfidf_words(tfidf_vectorizer, cluster_docs, feature_names, n_words=10):
    """Получение топ-N слов для TF-IDF"""
    cluster_vectors = tfidf_vectorizer.transform(cluster_docs)
    cluster_mean = np.mean(cluster_vectors.toarray(), axis=0)
    top_indices = np.argsort(cluster_mean)[-n_words:][::-1]
    return [feature_names[i] for i in top_indices]


def main():
    st.set_page_config(page_title="Text Clustering Analysis", layout="wide")
    st.title("Анализ кластеризации текстов")

    # Загрузка данных и моделей
    with st.spinner("Загрузка моделей и данных..."):
        models = load_models()
        corpus = load_corpus()

    st.sidebar.header("Настройки кластеризации")

    # Выбор количества документов для анализа
    sample_size = st.sidebar.slider(
        "Количество документов для анализа",
        min_value=100,
        max_value=len(corpus),
        value=min(1000, len(corpus)),
        step=100
    )

    corpus_sample = corpus[:sample_size]

    # Выбор методов
    tokenization_method = st.sidebar.selectbox(
        "Метод токенизации",
        ["Naive (whitespace)", "Regex", "BPE"]
    )

    vectorization_method = st.sidebar.selectbox(
        "Метод векторизации",
        ["TF-IDF", "Word2Vec", "FastText"]
    )

    clustering_method = st.sidebar.selectbox(
        "Алгоритм кластеризации",
        ["K-Means", "Mini-Batch K-Means", "Agglomerative", "Spectral", "HDBSCAN"]
    )

    n_clusters = st.sidebar.slider(
        "Количество кластеров",
        min_value=2,
        max_value=10,
        value=5,
        step=1
    )

    # Кнопка запуска анализа
    if st.sidebar.button("Запустить кластеризацию"):
        with st.spinner("Выполняется векторизация и кластеризация..."):
            # Векторизация
            embeddings, tokens_list, sentences, vectorizer_or_model = vectorize_text(
                corpus_sample, tokenization_method, vectorization_method, models
            )

            # Кластеризация
            if clustering_method == "K-Means":
                labels = k_means(embeddings, k=n_clusters)
            elif clustering_method == "Mini-Batch K-Means":
                labels = mini_batch_means(embeddings, n_clusters=n_clusters)
            elif clustering_method == "Agglomerative":
                labels = agglomerative_clustering(embeddings, n_clusters=n_clusters)
            elif clustering_method == "Spectral":
                labels = spectral_clustering(embeddings, n_clusters=n_clusters)
            else:  # HDBSCAN
                labels = use_hdbscan(embeddings)

            # Вычисление метрик
            metrics = get_metrics(embeddings, labels)

            # Визуализация
            st.header("Результаты кластеризации")

            # Метрики
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Silhouette Score",
                          f"{metrics['silhouette']:.3f}" if metrics['silhouette'] else "N/A")
            with col2:
                st.metric("Calinski-Harabasz",
                          f"{metrics['calinski_harabasz']:.3f}" if metrics['calinski_harabasz'] else "N/A")
            with col3:
                st.metric("Davies-Bouldin",
                          f"{metrics['davies_bouldin']:.3f}" if metrics['davies_bouldin'] else "N/A")

            # Визуализация кластеров
            st.subheader("Визуализация кластеров")

            # Уменьшение размерности для визуализации
            pca = PCA(n_components=2)
            embeddings_2d = pca.fit_transform(embeddings)

            viz_df = pd.DataFrame({
                'x': embeddings_2d[:, 0],
                'y': embeddings_2d[:, 1],
                'cluster': labels,
                'text': corpus_sample
            })

            fig = px.scatter(viz_df, x='x', y='y', color='cluster',
                             hover_data=['text'], title="PCA визуализация кластеров")
            st.plotly_chart(fig, use_container_width=True)

            # Анализ по кластерам
            st.subheader("Анализ по кластерам")

            unique_clusters = np.unique(labels)

            for cluster_id in unique_clusters:
                if cluster_id == -1:
                    continue

                cluster_mask = labels == cluster_id
                cluster_docs = [corpus_sample[i] for i in range(len(corpus_sample)) if cluster_mask[i]]
                cluster_size = len(cluster_docs)

                with st.expander(f"Кластер {cluster_id} (размер: {cluster_size})"):
                    # Топ слова для TF-IDF
                    if vectorization_method == "TF-IDF":
                        st.write("**Топ-10 характерных слов:**")
                        cluster_sentences = [sentences[i] for i in range(len(sentences)) if cluster_mask[i]]
                        if len(cluster_sentences) > 0:
                            cluster_vectors = vectorizer_or_model.transform(cluster_sentences)
                            cluster_mean = np.mean(cluster_vectors.toarray(), axis=0)
                            feature_names = vectorizer_or_model.get_feature_names_out()
                            top_indices = np.argsort(cluster_mean)[-10:][::-1]
                            top_words = [feature_names[i] for i in top_indices]
                            for word in top_words:
                                st.write(f"- {word}")

                    # Ближайшие слова для эмбеддингов
                    elif vectorization_method in ["Word2Vec", "FastText"]:
                        st.write("**Ближайшие слова к центроиду:**")
                        cluster_embeddings = embeddings[cluster_mask]
                        centroid = np.mean(cluster_embeddings, axis=0)
                        try:
                            similar_words = vectorizer_or_model.wv.most_similar(positive=[centroid], topn=10)
                            for word, similarity in similar_words:
                                st.write(f"- {word} (сходство: {similarity:.3f})")
                        except Exception as e:
                            st.error(f"Ошибка при поиске похожих слов: {e}")

            # Общая статистика
            st.subheader("Статистика кластеров")
            cluster_stats = pd.DataFrame({
                'Cluster': labels,
                'Count': 1
            }).groupby('Cluster').count().reset_index()

            fig_bar = px.bar(cluster_stats, x='Cluster', y='Count',
                             title="Распределение документов по кластерам")
            st.plotly_chart(fig_bar, use_container_width=True)


if __name__ == "__main__":
    main()