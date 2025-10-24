import numpy as np
from sklearn.decomposition import TruncatedSVD
import umap.umap_ as umap
import matplotlib.pyplot as plt
from classical_vectorizers import one_hot_matrix, one_hot_vectorizer, bow_matrix, bow_vectorizer, tfidf_vectorizer, tfidf_matrix

def apply_truncated_svd(X, preserve_variance=0.95, max_components=1000, random_state=0):
    X = np.asarray(X)
    n_samples, n_features = X.shape

    if max_components is None:
        max_components = min(n_samples, n_features)
    else:
        max_components = min(max_components, n_samples, n_features)


    fit_components = max(2, max_components)
    svd_full = TruncatedSVD(n_components=fit_components, random_state=random_state)
    svd_full.fit(X)

    explained = svd_full.explained_variance_ratio_

    cumulative = np.cumsum(explained)

    #ищем минимальное k с cumulative[k-1] >= preserve_variance
    idx = np.searchsorted(cumulative, preserve_variance, side='left')
    if idx < len(cumulative):
        chosen_k = idx + 1
    else:
        #не достигли требуемой дисперсии даже при fit_components
        chosen_k = fit_components

    #если выбранное k меньше, чем мы уже посчитали, обучим финальную модель с нужным k
    if chosen_k < fit_components:
        svd = TruncatedSVD(n_components=chosen_k, random_state=random_state)
        X_reduced = svd.fit_transform(X)
    else:
        svd = svd_full
        X_reduced = svd.transform(X)

    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(cumulative) + 1), cumulative, marker='o', linewidth=1)
    plt.xlabel('Число компонент')
    plt.ylabel('Накопленная объяснённая дисперсия')
    plt.title('Зависимость числа компонент от уровня сохраняемой дисперсии')
    plt.show()

    return svd, X_reduced

def hidden_thems(svd_obj, vectorizer, top_thems=10):
    count = 0
    for topic_idx, topic_weights in enumerate(svd_obj.components_):
        count += 1
        if count > top_thems:
            break
        top_indices = topic_weights.argsort()[::-1][:10]  # индексы самых важных слов
        top_words = [vectorizer.get_feature_names_out()[i] for i in top_indices]
        print(f"Тема {topic_idx + 1}: {top_words}")

def draw_hidden_thems(x_red):
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0)
    X_umap = reducer.fit_transform(x_red)
    plt.scatter(X_umap[:, 0], X_umap[:, 1], s=5, alpha=0.7)
    plt.title("Тексты в пространстве скрытых семантических тем (UMAP)")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.grid(True)
    plt.show()

svd_onehot, X_onehot_red = apply_truncated_svd(one_hot_matrix.toarray(), preserve_variance=0.90, max_components=300)
draw_hidden_thems(X_onehot_red)
hidden_thems(svd_onehot, one_hot_vectorizer)
print("-----------------------------------")
svd_bow, X_bow_red = apply_truncated_svd(bow_matrix.toarray(), preserve_variance=0.90, max_components=300)
draw_hidden_thems(X_bow_red)
hidden_thems(svd_bow, bow_vectorizer)
print("-----------------------------------")
svd_tfidf, X_tfidf_red = apply_truncated_svd(tfidf_matrix.toarray(), preserve_variance=0.90, max_components=300)
draw_hidden_thems(X_tfidf_red)
hidden_thems(svd_tfidf, tfidf_vectorizer)