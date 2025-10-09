"""
3.4. Этап 4. Сравнительный анализ методов токенизации и нормализации
Задача: Провести всестороннее эмпирическое сравнение эффективности различных методов.

План эксперимента:
Методы:

Токенизация:
Наивная (по пробелам)
На основе регулярных выражений
Библиотеки: nltk, spacy, razdel (для русского — особенно рекомендуется)
Стемминг:
PorterStemmer, SnowballStemmer (русский)
Лемматизация:
pymorphy2, spacy (ru_core_news_sm)
Метрики оценки:

Объём словаря — количество уникальных токенов (меньше → компактнее)
Доля OOV (Out-of-Vocabulary) — % слов, не вошедших в словарь (ниже → лучше обобщение)
Скорость обработки — время на 1000 статей (важно для production)
Семантическая согласованность — сохраняется ли смысл?
→ Можно оценить через косинусное сходство эмбеддингов до/после обработки или экспертную оценку на выборке.
Оформление результатов:
Сведите результаты в сводную таблицу tokenization_metrics.csv и добавьте анализ в отчёт.
"""
import json
import re
import time
import csv
from typing import List, Callable, Tuple
import numpy as np
import razdel
import nltk
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---- Utilities ----

RE_CYRILLIC_WORD = re.compile(r"[A-Za-zА-Яа-яЁё]+(?:[-'][A-Za-zА-Яа-яЁё]+)*", flags=re.UNICODE)

"""
возвращает обработанные тексты из jsonl файла.
параметры:
path - путь к jsonl файлу
max_records - максимальное количество текстов, которое мы хотим считать
"""
def read_jsonl_texts(path: str, max_records: int = None) -> List[str]:
    texts = []
    text_field = "text"
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_records and i >= max_records:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if text_field in obj and obj[text_field]:
                texts.append(obj[text_field])
    return texts

# ---- токенизаторы ----

def tokenize_naive(text: str) -> List[str]:
    # Простая токенизация по пробелам (и отделяем лишние пунктуации у концов)
    # split by whitespace and strip punctuation from ends
    parts = text.split()
    tokens = [p.strip("«»()[]{}.,:;!?\"'“”—–…") for p in parts if p.strip("«»()[]{}.,:;!?\"'“”—–…")]
    return tokens

def tokenize_regex(text: str) -> List[str]:
    return RE_CYRILLIC_WORD.findall(text)

def tokenize_razdel(text: str) -> List[str]:
    return [t.text for t in razdel.tokenize(text)]

def tokenize_nltk(text: str) -> List[str]:
    try:
        from nltk import word_tokenize
    except Exception:
        nltk.download("punkt")
        from nltk import word_tokenize
    return word_tokenize(text)

def tokenize_spacy(text: str, nlp) -> List[str]:
    doc = nlp(text)
    return [t.text for t in doc]

# ---- Нормализаторы (стемминг/лемматизация) ----

class Normalizer:
    def __init__(self, method: str, spacy_nlp=None):
        self.method = method  # 'none', 'porter', 'snowball', 'pymorphy', 'spacy_lemma'
        self.spacy_nlp = spacy_nlp
        self._init_resources()

    def _init_resources(self):
        self.porter = None
        self.snowball = None
        self.morph = None
        if self.method == "porter":
            from nltk.stem import PorterStemmer
            self.porter = PorterStemmer()
        elif self.method == "snowball":
            from nltk.stem.snowball import SnowballStemmer
            self.snowball = SnowballStemmer("russian")

    def normalize_token(self, token: str) -> str:
        if self.method == "none":
            return token
        if self.method == "porter":
            return self.porter.stem(token)
        if self.method == "snowball":
            return self.snowball.stem(token)
        if self.method == "spacy_lemma":
            return token
        return token

    def normalize_tokens(self, tokens: List[str]) -> List[str]:
        if self.method == "spacy_lemma":
            doc = self.spacy_nlp(" ".join(tokens))
            return [t.lemma_ if t.lemma_ else t.text for t in doc]
        else:
            return [self.normalize_token(t) for t in tokens]

# ---- метрики ----

def vocab_size(tokens_list: List[List[str]]) -> int:
    s = set()
    for toks in tokens_list:
        s.update([t.lower() for t in toks if t])
    return len(s)

def compute_oov_rate(train_tokens: List[List[str]], test_tokens: List[List[str]]) -> float:
    train_vocab = set(t.lower() for toks in train_tokens for t in toks)
    n_total = 0
    n_oov = 0
    for toks in test_tokens:
        for t in toks:
            n_total += 1
            if t.lower() not in train_vocab:
                n_oov += 1
    return 0.0 if n_total == 0 else (n_oov / n_total) * 100.0

# ---- косинусное сходство на ембедингах ----

class Embedder:
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        return np.vstack(self.model.encode(texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True))

def semantic_similarity_original_vs_processed(original_texts: List[str], processed_texts: List[str], sample_size: int = 200) -> float:
    """
    Вычислить среднее косинусное сходство между исходными текстами и обработанными текстами
    sample_size - сколько пар нужно выбрать (случайным образом), чтобы вычислить среднее значение скорости.
    """
    n = min(len(original_texts), len(processed_texts))
    if n == 0:
        return 0.0
    idxs = np.random.choice(n, min(sample_size, n), replace=False)
    embedder = Embedder()
    originals = [original_texts[i] for i in idxs]
    procs = [processed_texts[i] for i in idxs]
    emb_orig = embedder.embed_texts(originals)
    emb_proc = embedder.embed_texts(procs)
    sim_matrix = cosine_similarity(emb_orig, emb_proc)
    sims = np.diag(sim_matrix)
    return float(np.mean(sims))

# ---- измерение времени на токенизацию + нормализацию ----

def time_processing(fn_tokenize: Callable[[str], List[str]], normalizer: Normalizer, texts: List[str], n_samples: int = 1000) -> Tuple[float, List[List[str]]]:
    """
    измерение времени на токенизацию + нормализацию n_samples текстов
    Возвращает (продолжительность, список со списками нормализованных токенов)
    """
    n = min(len(texts), n_samples)
    start = time.time()
    toks_all = []
    for text in texts[:n]:
        toks = fn_tokenize(text)
        toks_norm = normalizer.normalize_tokens(toks)
        toks_all.append(toks_norm)
    duration = time.time() - start
    return duration, toks_all

# ---- исследование ----

def run_experiment(
    texts: List[str],
    results_csv: str = "tokenization_metrics.csv",
    sample_for_speed: int = 1000,
    sample_for_semantic: int = 200,
    max_texts_for_experiment: int = None,
):
    #Опционально разделяем корпус для тестирования скорости, сохраняя при этом разнообразие
    if max_texts_for_experiment:
        texts = texts[:max_texts_for_experiment]

    n_texts = len(texts)
    print(f"Всего текстов: {n_texts}")

    #обучение/тест разделяем для Out-of-Vocabulary (80/20)
    split_idx = int(0.8 * n_texts)
    train_texts = texts[:split_idx]
    test_texts = texts[split_idx:]

    spacy_nlp = None
    #попытка загрузить ру модель
    try:
        spacy_nlp = spacy.load("ru_core_news_sm")
    except Exception:
        pass

    #подготовка токенизаторов
    tokenizers = {
        "naive": lambda text: tokenize_naive(text),
        "regex": lambda text: tokenize_regex(text),
        "nltk": lambda text: tokenize_nltk(text),
        "razdel": lambda text: tokenize_razdel(text),
        "spacy":  lambda text: tokenize_spacy(text, spacy_nlp)
    }

    #методы нормализации для проверки
    normalizer_names = ["none", "porter", "snowball", "spacy_lemma"]
    avail_normalizers = []
    for name in normalizer_names:
        try:
            #Тестовая инициализация
            Normalizer(name, spacy_nlp)
            avail_normalizers.append(name)
        except Exception as e:
            print(f"Пропущен нормализатор {name}: {e}")

    print(f"ТОкенизаторы для оценки: {list(tokenizers.keys())}")
    print(f"Нормализаторы для оценки: {avail_normalizers}")

    #Мы будем оценивать каждый токенизатор + каждый нормализатор из normalizer_names
    results = []
    #Сохраняйте обработанные тексты для семантической оценки: оригиналы и обработанное объединение для каждого метода.
    processed_texts_cache = {}  #ключ -> список обработанных текстовых строк (соединенных токенов)

    #предварительно вычислить исходные тексты встраивания в виде списка строк
    originals = texts

    #для каждого токенизатора
    for tk_name, tk_fn in tokenizers.items():
        #для каждого нормализатора
        for norm_name in avail_normalizers:
            #создаём функциб для нормализации
            normalizer = Normalizer(norm_name, spacy_nlp)
            #измерить скорость на sample_for_speed
            duration, toks_sample = time_processing(tk_fn, normalizer, texts, n_samples=sample_for_speed)
            #разбить токенизированный и нормализованный текст на обучение + тест для подсчёта oov
            train_tokens = toks_sample[:split_idx]
            test_tokens = toks_sample[split_idx:]

            base_vocab_size = vocab_size(train_tokens)#считаем размер словаря на тренировочной выборке
            base_oov = compute_oov_rate(train_tokens, test_tokens)
            # семантическое сходство: присоединяйте токены обратно к тексту и вычисляйте сходство встраивания
            processed_texts_joined = [" ".join([tok for tok in t]) for t in toks_sample]
            processed_texts_cache[(tk_name, norm_name)] = processed_texts_joined
            try:
                sem_sim = semantic_similarity_original_vs_processed(originals, processed_texts_joined, sample_size=sample_for_semantic)
            except Exception as e:
                print("Пропуск метрики семантического сходства:", e)
                sem_sim = None

            results.append({
                "tokenizer": tk_name,
                "normalizer": norm_name,
                "vocab_size": base_vocab_size,
                "oov_pct": base_oov,
                f"time_seconds_per_{min(len(texts), sample_for_speed)}_samples": duration,
                "semantic_similarity": sem_sim,
            })

    # Сохранение в CSV
    fieldnames = ["tokenizer", "normalizer", "vocab_size", "oov_pct", f"time_seconds_per_{min(len(texts), sample_for_speed)}_samples", "semantic_similarity"]
    with open(results_csv, "w", encoding="utf-8", newline='') as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {
                "tokenizer": r.get("tokenizer"),
                "normalizer": r.get("normalizer"),
                "vocab_size": r.get("vocab_size"),
                "oov_pct": f"{r.get('oov_pct'):.4f}",
                f"time_seconds_per_{min(len(texts), sample_for_speed)}_samples": f"{r.get(f'time_seconds_per_{min(len(texts), sample_for_speed)}_samples'):.4f}",
                "semantic_similarity": (f"{r.get('semantic_similarity'):.4f}"),
            }
            writer.writerow(row)

    print(f"Результаты сохранены в {results_csv}")


def main():
    sample = 100
    input_path = "processed.jsonl"
    output_csv = "tokenization_metrics.csv"
    speed_sample = 100
    semantic_sample = 25


    texts = read_jsonl_texts(input_path, max_records=sample)
    if not texts:
        print("В json файле не найден текст")
        return
    run_experiment(
        texts=texts,
        results_csv=output_csv,
        sample_for_speed=speed_sample,
        sample_for_semantic=semantic_sample,
        max_texts_for_experiment=sample
    )

if __name__ == "__main__":
    main()
