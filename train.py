"""
3.5. Этап 5. Обучение подсловных моделей токенизации
Задача: Обучить три подсловные модели на едином корпусе и провести их сравнительный анализ.

Указания к выполнению:
Модели:

Byte Pair Encoding (BPE)
WordPiece
Unigram Language Model
Инструменты:
Рекомендуется использовать: tokenizers (Hugging Face) или sentencepiece

Параметры обучения:

Размер словаря: 8 000 – 32 000 токенов (обязательно протестируйте несколько значений)
Минимальная частота токена: 2–5
Метрики оценки:

Процент фрагментации слов — доля слов, разбитых на 2+ подслова (показывает «агрессивность» модели)
Коэффициент сжатия — отношение числа исходных слов к числу токенов после обработки
Эффективность реконструкции — насколько точно модель восстанавливает исходный текст
"""

import json
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece, Unigram
from tokenizers.trainers import BpeTrainer, WordPieceTrainer, UnigramTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Sequence, NFKC

import pandas as pd
from rapidfuzz.distance import Levenshtein as RFLevenshtein

def normalized_edit_distance(a: str, b: str) -> float:
    #возвращает нормализованное расстояние в [0,1], 0 означает идентичное, 1 означает совершенно другое
    if len(a) == 0 and len(b) == 0:
        return 0.0
    d = RFLevenshtein.normalized_distance(a, b)  # уже 0..1
    return float(d)


def load_texts_from_jsonl(path: str, text_field: str = "text") -> List[str]:
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                t = obj.get(text_field, "")
                if t is None:
                    continue
                t = t.strip()
                if t:
                    texts.append(t)
            except Exception:
                continue
    return texts


def dump_plain_corpus(texts: List[str], out_path: str):
    #записываем в обычный текстовый файл с одним текстом в строке
    with open(out_path, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(t.replace("\n", " ") + "\n")


def train_bpe(files: List[str], vocab_size: int, min_frequency: int, out_path: str) -> Tokenizer:
    tok = Tokenizer(BPE(unk_token="[UNK]"))
    tok.normalizer = Sequence([NFKC()])  # юникод нормализация
    tok.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["[UNK]"]
    )
    tok.train(files, trainer)
    tok.save(out_path)
    return Tokenizer.from_file(out_path)


def train_wordpiece(files: List[str], vocab_size: int, min_frequency: int, out_path: str) -> Tokenizer:
    tok = Tokenizer(WordPiece(unk_token="[UNK]"))
    tok.normalizer = Sequence([NFKC()])
    tok.pre_tokenizer = Whitespace()
    trainer = WordPieceTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["[UNK]"],
    )
    tok.train(files, trainer)
    tok.save(out_path)
    return Tokenizer.from_file(out_path)


def train_unigram(files: List[str], vocab_size: int, out_path: str) -> Tokenizer:
    tok = Tokenizer(Unigram())
    tok.normalizer = Sequence([NFKC()])
    tok.pre_tokenizer = Whitespace()
    trainer = UnigramTrainer(
        vocab_size=vocab_size,
        unk_token="[UNK]"
    )
    tok.train(files, trainer)
    tok.save(out_path)
    return Tokenizer.from_file(out_path)


def evaluate_tokenizer_on_texts(tok: Tokenizer, texts: List[str]) -> Dict:
    """
    Вычислить:
      - всего слов
      - всего токенов
      - фрагментированные слова (слова, которые разбиты на 2+ подслова)
      - фрагментация процент
      - коэффициент сжатия (слова/токены)
      - Степень соответствия между исходными и восстановленными данными
      - качество реконструкции текста (0..1)
    """
    total_words = 0
    total_tokens = 0
    fragmented = 0
    exact_matches = 0
    total_lines = 0
    edit_dist_sum = 0.0

    for line in tqdm(texts, desc="Evaluating", leave=False):
        total_lines += 1
        words = line.split()
        total_words += len(words)

        # фрагментация по словам и подсчет токенов
        for w in words:
            try:
                enc = tok.encode(w)
                toks = enc.tokens
            except Exception:
                #рассматривать как один токен
                toks = [w]
            total_tokens += max(1, len(toks))
            if len(toks) > 1:
                fragmented += 1

        #реконструкция: кодировать-декодировать всю строку
        try:
            enc_line = tok.encode(line)
            decoded = tok.decode(enc_line.ids)
        except Exception:
            #если декодирование не удалось, рассматривать как несовпадающее с максимальным расстоянием
            decoded = ""

        #прямое точное совпадение
        if decoded == line:
            exact_matches += 1

        edit_dist_sum += normalized_edit_distance(line, decoded)

    frag_percent = (fragmented / total_words * 100.0) if total_words > 0 else 0.0
    compression = (total_words / total_tokens) if total_tokens > 0 else 0.0
    exact_frac = (exact_matches / total_lines) if total_lines > 0 else 0.0
    avg_norm_edit = (edit_dist_sum / total_lines) if total_lines > 0 else 1.0

    return {
        "total_lines": total_lines,
        "total_words": total_words,
        "total_tokens": total_tokens,
        "fragmented_words": fragmented,
        "frag_percent": frag_percent,
        "compression_ratio": compression,
        "reconstruction_exact_frac": exact_frac,
        "avg_normalized_edit_distance": avg_norm_edit,
    }


def main():
    corpus_path = "corpus.jsonl"
    outdir = Path("./output")
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading corpus from", corpus_path)
    texts = load_texts_from_jsonl(corpus_path, text_field="text")
    if len(texts) == 0:
        raise SystemExit("No texts found in corpus. Check JSONL and field name.")

    print(f"Loaded {len(texts)} documents. Example length (chars):", len(texts[0]))

    #создать простой текстовый файл для тренера
    plain_corpus_path = outdir / "corpus_for_training.txt"
    dump_plain_corpus(texts, str(plain_corpus_path))

    vocab_sizes = [8000, 16000, 32000]
    min_freq = 3

    #запишем результаты в список, а затем в DataFrame.
    results = []

    #сопоставление имени -> функции обучения
    training_map = {
        "bpe": train_bpe,
        "wordpiece": train_wordpiece,
        "unigram": train_unigram,
    }

    files_for_train = [str(plain_corpus_path)]

    for model_name, train_fn in training_map.items():
        for vocab_size in vocab_sizes:
            #построить выходные имена
            model_fname = f"{model_name}_v{vocab_size}_minf{min_freq}.json"
            model_path = outdir / model_fname
            print(f"\nТренируется {model_name.upper()} | словарь={vocab_size} | последовательность={min_freq}")

            try:
                tok = train_fn(files_for_train, vocab_size=vocab_size, min_frequency=min_freq, out_path=str(model_path))
            except TypeError as e:
                print("Ошибка:", e)
                #попытка без min_frequency
                if model_name == "unigram":
                    tok = train_unigram(files_for_train, vocab_size=vocab_size, out_path=str(model_path))

            eval_stats = evaluate_tokenizer_on_texts(tok, texts)

            #сохразяем строку со статистикой
            row = {
                "model": model_name,
                "vocab_size": vocab_size,
                "min_frequency": min_freq,
                "model_file": str(model_path),
            }
            row.update(eval_stats)
            results.append(row)

            print(f"frag%: {eval_stats['frag_percent']:.3f} | compression: {eval_stats['compression_ratio']:.3f} | exact_recon_frac: {eval_stats['reconstruction_exact_frac']:.3f} | avg_norm_edit: {eval_stats['avg_normalized_edit_distance']:.3f}")

    #сохраняем в csv
    df = pd.DataFrame(results)
    csv_path = outdir / "training_results.csv"
    df.to_csv(csv_path, index=False)
    print("Конец обучения")


if __name__ == "__main__":
    main()
