from gensim.models import Word2Vec, FastText, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.utils import simple_preprocess
from scipy.stats import spearmanr
import time, tracemalloc
import json

SKIPGRAM = "skipgram"
CBOW = "sbow"
PVDM = "pv-dm"
PVDBOW = "pv-dbow"
VECTOR_SIZES = [100, 200, 300]
WINDOW_SIZES = [5, 8, 10]
MIN_COUNTS = [5, 7, 10]
#корпус
corpus_filename = "processed_corpus.jsonl"
corpus = []
with open(corpus_filename, "r", encoding="utf-8") as infile:
    for line in infile:
        data = json.loads(line)
        if "text" in data:
            corpus.append(data["text"])
test_corpus = corpus[int(len(corpus) * 0.1) :]
corpus = corpus[: int(len(corpus) * 0.1)]
#токенизация для word-level
sentences = [simple_preprocess(doc) for doc in corpus]
test_sentences = [simple_preprocess(doc) for doc in test_corpus]

#подготовка для document-level. Каждому доку нужен уникльный тег
tagged_docs = [
    TaggedDocument(words=simple_preprocess(doc), tags=[f"DOC_{i}"])
    for i, doc in enumerate(corpus)
]

def measure_mem(model, type):
    if type == SKIPGRAM or type == CBOW:
        return model.wv.vectors.shape[0] * model.wv.vectors.shape[1] * 4 / (2**10) #мб
    elif type == PVDM or type == PVDBOW:
        return model.dv.vectors.shape[0] * model.dv.vectors.shape[1] * 4 / (2**10)#мб
    return None


def measure_time(func, *args, **kwargs):
    tracemalloc.start()
    t0 = time.perf_counter()
    res = func(*args, **kwargs)
    t1 = time.perf_counter()
    return res, t1 - t0

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

#метрики
#кореляция с человеческими оценками
def human_grade(model):
    # пары (слово1, слово2, человеческая_оценка)
    pairs = [
        ("москва", "россия", 9.5),
        ("париж", "франция", 9.4),
        ("киев", "украина", 9.5),
        ("дом", "книга", 1.0),
        ("кошка", "собака", 4.1)
    ]
    # фильтруем только те пары, где оба слова есть в словаре
    valid = [(a, b, s) for a, b, s in pairs if a in model.wv and b in model.wv]

    # вычисляем сходство модели
    model_scores = [model.wv.similarity(a, b) for a, b, _ in valid]
    human_scores = [s for _, _, s in valid]
    #корреляция с человеческими оценками
    corr, p = spearmanr(model_scores, human_scores)
    return corr#0 кореляции нет. 1 - кореляция есть, -1 обратная кореляция

def ovv(model, test_sentences):
    # все токены в один список
    all_tokens = [word for sent in test_sentences for word in sent]
    # уникальные слова
    vocab = set(all_tokens)

    oov = [w for w in vocab if w not in model.wv]

    oov_ratio = len(oov) / len(vocab)
    return oov_ratio



m = get_word_2_vec(sentences, VECTOR_SIZES[0], WINDOW_SIZES[0], MIN_COUNTS[0], SKIPGRAM)
print(m.wv.vectors.shape)
results = []
for vector in VECTOR_SIZES:
    for window in WINDOW_SIZES:
        for min_count in MIN_COUNTS:
            print(f"Вектор {vector}, окно {window}, минимальная частота слов{min_count}")
            print("Модель word2vec версия skipgram")
            model, t = measure_time(get_word_2_vec, sentences, vector, window, min_count, SKIPGRAM)
            weight = measure_mem(model, SKIPGRAM)
            corr = human_grade(model)
            oov_ratio = ovv(model, test_sentences)
            print(f"Время выполнения {t} мс, вес модели {weight} мб, корреляция с человеческими оценками {corr}, где 1 - полная коррелирование, 0 - его отсутствие, -1 - обратное коррелирование. Доля слов вне словаря - {oov_ratio}")

            print("Модель word2vec версия cbow")
            model, t = measure_time(get_word_2_vec, sentences, vector, window, min_count, CBOW)
            weight = measure_mem(model, CBOW)
            corr = human_grade(model)
            oov_ratio = ovv(model, test_sentences)
            print(
                f"Время выполнения {t} мс, вес модели {weight} мб, "
                f"корреляция с человеческими оценками {corr}, где 1 - полная коррелирование, 0 - его отсутствие, -1 - обратное коррелирование. "
                f"Доля слов вне словаря - {oov_ratio}")

            print("Модель FastText версия skipgram")
            model, t = measure_time(get_fast_text, sentences, vector, window, min_count, SKIPGRAM)
            weight = measure_mem(model, SKIPGRAM)
            corr = human_grade(model)
            oov_ratio = ovv(model, test_sentences)
            print(
                f"Время выполнения {t} мс, вес модели {weight} мб, "
                f"корреляция с человеческими оценками {corr}, где 1 - полная коррелирование, 0 - его отсутствие, -1 - обратное коррелирование. "
                f"Доля слов вне словаря - {oov_ratio}")

            print("Модель FastText версия cbow")
            model, t = measure_time(get_fast_text, sentences, vector, window, min_count, CBOW)
            weight = measure_mem(model, CBOW)
            corr = human_grade(model)
            oov_ratio = ovv(model, test_sentences)
            print(
                f"Время выполнения {t} мс, вес модели {weight} мб, "
                f"корреляция с человеческими оценками {corr}, где 1 - полная коррелирование, 0 - его отсутствие, -1 - обратное коррелирование. "
                f"Доля слов вне словаря - {oov_ratio}")

            print("Модель doc2vec версия pv-dm")
            model, t = measure_time(get_doc_2_vec, tagged_docs, vector, window, min_count, PVDM, 20)
            weight = measure_mem(model, PVDM)
            print(f"Время выполнения {t} мс, вес модели {weight} мб")

            print("Модель doc2vec версия pv-dbow")
            model, t = measure_time(get_doc_2_vec, tagged_docs, vector, window, min_count, PVDBOW, 20)
            weight = measure_mem(model, PVDBOW)
            print(f"Время выполнения {t} мс, вес модели {weight} мб")