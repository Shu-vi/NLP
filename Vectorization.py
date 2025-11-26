from gensim.models import Word2Vec, FastText
import joblib

def create_tfidf():
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    def _inner(docs):
        return vectorizer.transform(docs).toarray()
    return _inner

def create_w2v():
    model = Word2Vec.load("./word2vec.model")
    def _inner(word):
        if word in model.wv:
            return model.wv[word]
        else:
            return None
    return _inner

def create_fasttext():
    model = FastText.load("./fasttext.model")
    def _inner(word):
        if word in model.wv:
            return model.wv[word]
        else:
            return None
    return _inner