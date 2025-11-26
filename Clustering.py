from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity
import hdbscan


def k_means(docs, k = 5):
    return KMeans(n_clusters=k).fit_predict(docs)

def mini_batch_means(docs, n_clusters):
    model = MiniBatchKMeans(n_clusters=n_clusters, batch_size=256, random_state=42)
    return model.fit_predict(docs)

def use_hdbscan(docs):
    model = hdbscan.HDBSCAN(metric="euclidean", min_cluster_size=3)
    return model.fit_predict(docs)

def agglomerative_clustering(docs, n_clusters=5):
    model = AgglomerativeClustering(n_clusters=n_clusters, metric="cosine", linkage="average")
    return model.fit_predict(docs)

def spectral_clustering(docs, n_clusters=5):
    sim = cosine_similarity(docs)
    model = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        random_state=42
    )
    return model.fit_predict(sim)