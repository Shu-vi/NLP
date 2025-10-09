"""
Запуск streamlit run app_streamlit.py
"""
import json
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

import streamlit as st
import pandas as pd
import plotly.express as px

from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece, Unigram
from tokenizers.trainers import BpeTrainer, WordPieceTrainer, UnigramTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Sequence, NFKC
from rapidfuzz.distance import Levenshtein as RFLevenshtein

def normalized_distance(a: str, b: str) -> float:
    if not a and not b:
        return 0.0
    return float(RFLevenshtein.normalized_distance(a, b))


# -------------------------
# Загрузка корпуса
# -------------------------
def load_jsonl_texts(file_bytes: bytes, text_field: str = "text", max_docs: int = None) -> List[str]:
    """
    Извлекаем текст для обучения из JSONL файла
    """
    texts: List[str] = []
    try:
        s = file_bytes.decode("utf-8")
    except Exception:
        s = file_bytes.decode("utf-8", errors="replace")
    for line in s.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            t = obj.get(text_field, "")
            if t is None:
                continue
            t = str(t).strip()
            if t:
                texts.append(t)
        except Exception:
            # игнорируем некорректные строки
            continue
        if max_docs and len(texts) >= max_docs:
            break
    return texts


def jsonl_bytes_to_textfile(file_bytes: bytes, out_path: str, text_field: str = "text"):
    texts = load_jsonl_texts(file_bytes, text_field=text_field)
    with open(out_path, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(t.replace("\n", " ") + "\n")
    return texts


# -------------------------
# Тренировка токенизаторов
# -------------------------
def train_bpe(filepaths: List[str], vocab_size: int = 16000, min_freq: int = 2, unk_token: str = "[UNK]") -> Tokenizer:
    tok = Tokenizer(BPE(unk_token=unk_token))
    tok.normalizer = Sequence([NFKC()])
    tok.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=vocab_size, min_frequency=min_freq, special_tokens=[unk_token])
    tok.train(filepaths, trainer)
    return tok


def train_wordpiece(filepaths: List[str], vocab_size: int = 16000, min_freq: int = 2, unk_token: str = "[UNK]") -> Tokenizer:
    tok = Tokenizer(WordPiece(unk_token=unk_token))
    tok.normalizer = Sequence([NFKC()])
    tok.pre_tokenizer = Whitespace()
    trainer = WordPieceTrainer(vocab_size=vocab_size, min_frequency=min_freq, special_tokens=[unk_token])
    tok.train(filepaths, trainer)
    return tok


def train_unigram(filepaths: List[str], vocab_size: int = 16000) -> Tokenizer:
    tok = Tokenizer(Unigram())
    tok.normalizer = Sequence([NFKC()])
    tok.pre_tokenizer = Whitespace()
    trainer = UnigramTrainer(
        vocab_size=vocab_size,
        unk_token="[UNK]"
    )
    tok.train(filepaths, trainer)
    return tok


# -------------------------
# токенизация и метрики
# -------------------------
def tokenize_texts(tok: Tokenizer, texts: List[str]) -> Tuple[List[List[str]], List[List[int]]]:
    """
    Для каждой строки возвращает:
      - список токенов для каждого текста
      - список идентификаторов токенов для каждого текста
    """
    tokens_per_line = []
    ids_per_line = []
    for line in texts:
        enc = tok.encode(line)
        tokens_per_line.append(enc.tokens)
        ids_per_line.append(enc.ids)
    return tokens_per_line, ids_per_line


def compute_token_statistics(tok: Tokenizer, texts: List[str], top_n: int = 30) -> Dict:
    """
    Вычисляем:
      - частоту токена (глобальная)
      - распределение токенов на слово
      - распределение длины токена
      - Коэффициент OOV (счёт токенов равен unk)
    """
    total_tokens = 0
    unk_count = 0
    token_freq = {}

    tokens_per_word_counts = []
    token_lengths = []

    for line in texts:
        enc_line = tok.encode(line)
        toks = enc_line.tokens
        ids = enc_line.ids
        total_tokens += len(ids)
        for t in toks:
            token_freq[t] = token_freq.get(t, 0) + 1
            token_lengths.append(len(t))
            if t == "[UNK]" or t == "[unk]":
                unk_count += 1

        words = line.split()
        for w in words:
            enc_w = tok.encode(w)
            toks_w = enc_w.tokens
            tokens_per_word_counts.append(len(toks_w))

    # защита от деления на ноль
    oov_ratio = (unk_count / total_tokens) if total_tokens > 0 else 0.0

    freq_items = sorted(token_freq.items(), key=lambda x: x[1], reverse=True)
    top_tokens = freq_items[:top_n]

    stats = {
        "total_tokens": total_tokens,
        "unk_count": unk_count,
        "oov_ratio": oov_ratio,
        "top_tokens": top_tokens,
        "tokens_per_word_counts": tokens_per_word_counts,
        "token_lengths": token_lengths,
        "token_freq_series": pd.Series(token_freq),
        "tokens_per_word_mean": float(np.mean(tokens_per_word_counts)) if tokens_per_word_counts else 0.0,
        "tokens_per_word_median": float(np.median(tokens_per_word_counts)) if tokens_per_word_counts else 0.0,
    }
    return stats


# -------------------------
# экспортируем статистику (в html)
# -------------------------
def build_html_report(texts: List[str], tok: Tokenizer, stats: Dict, title: str = "Отчёт по токенизации") -> str:
    # создаём DataFrame самых частых токенов
    top_tokens = stats.get("top_tokens", [])
    df_top = pd.DataFrame(top_tokens, columns=["token", "count"])

    # создаём фигуры Plotly
    fig_len = px.histogram(stats.get("token_lengths", []), nbins=40, labels={"value": "Длина токена (симв.)"},
                           title="Распределение длины токена")
    fig_tpw = px.histogram(stats.get("tokens_per_word_counts", []), nbins=20, labels={"value": "Подслов на слово"},
                           title="Распределение токенов на слово")
    fig_top = px.bar(df_top.head(50), x="token", y="count", title="Частовстречаемые токены (топ)")

    html_parts = []
    html_parts.append(f"<h1>{title}</h1>")
    html_parts.append(f"<p>Всего текстов: {len(texts)}; Всего токенов: {stats.get('total_tokens', 0)}; OOV: {stats.get('oov_ratio', 0.0):.4f}</p>")
    html_parts.append("<h2>Самые частовстречаемые токены</h2>")
    html_parts.append(df_top.to_html(index=False))
    html_parts.append("<h2>Графики</h2>")
    # Подключаем plotly js в первом графике (cdn), остальные вставляем без повторного include
    html_parts.append(fig_len.to_html(full_html=False, include_plotlyjs="cdn"))
    html_parts.append(fig_tpw.to_html(full_html=False, include_plotlyjs=False))
    html_parts.append(fig_top.to_html(full_html=False, include_plotlyjs=False))

    return "\n".join(html_parts)


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Токенизатор", layout="wide")
st.title("Токенизатор - интерактивный анализ (JSONL)")

st.markdown(
    """
    Прототип веб-интерфейса для интерактивного анализа токенизаторов (BPE / WordPiece / Unigram) с использованием tokenizers.
    Формат корпуса: JSONL (каждая строка - JSON с полем text, которое хранит данные для обучения).
    """
)

# боковая панель
with st.sidebar:
    st.header("Корпус & модель")
    uploaded = st.file_uploader("Загрузите corpus.jsonl (JSONL, поле 'text')", type=["jsonl", "json"], accept_multiple_files=False)

    text_field = st.text_input("JSON с полем text", value="text")
    max_docs = st.number_input("Максимум загруженных документов (0=все)", min_value=0, step=1, value=0)
    st.markdown("---")
    st.subheader("Токенизатор")
    model_choice = st.selectbox("Выберите модель", ["BPE", "WordPiece", "Unigram"])
    vocab_size = st.selectbox("Размер словаря", [8000, 16000, 32000], index=2)
    min_freq = st.selectbox("min_frequency (BPE/WordPiece)", [2, 3, 4, 5], index=1)
    st.markdown("Unigram: min_frequency игнорируется")
    st.markdown("---")
    st.write("Можно загрузить готовый tokenizer JSON (tokenizers .json)")
    uploaded_tokenizer = st.file_uploader("Загрузить tokenizer .json (опционально)", type=["json"], accept_multiple_files=False)
    st.markdown("---")
    st.write("Экспорт")
    export_html_name = st.text_input("Имя HTML отчёта", value="tokenizer_report.html")

# --- Инициализация session_state ---
if "tokenizer_obj" not in st.session_state:
    st.session_state.tokenizer_obj = None
if "stats" not in st.session_state:
    st.session_state.stats = None
if "texts" not in st.session_state:
    st.session_state.texts = []

# загружаем корпус, если файл выбран
if uploaded is not None:
    try:
        raw = uploaded.getvalue()
        # преобразование в список текстов и сохранение в st.session_state
        st.session_state.texts = load_jsonl_texts(raw, text_field=text_field, max_docs=(None if max_docs == 0 else max_docs))
        # также создаём файл для тренера (если потребуется тренировать)
        tmp_dir = tempfile.mkdtemp()
        corpus_txt = Path(tmp_dir) / "corpus_for_training.txt"
        with open(corpus_txt, "w", encoding="utf-8") as f:
            for t in st.session_state.texts:
                f.write(t.replace("\n", " ") + "\n")
        st.success(f"Загружено документов: {len(st.session_state.texts)} (поле '{text_field}')")
    except Exception as e:
        st.error("Ошибка чтения jsonl: " + str(e))
else:
    st.info("Загрузите corpus.jsonl в сайдбаре")

# загрузка внешнего tokenizer
if uploaded_tokenizer is not None:
    try:
        tmp = uploaded_tokenizer.getvalue()
        tmp_path = Path(tempfile.mkdtemp()) / "uploaded_tok.json"
        with open(tmp_path, "wb") as f:
            f.write(tmp)
        st.session_state.tokenizer_obj = Tokenizer.from_file(str(tmp_path))
        st.success("Загружен внешний tokenizer.")
    except Exception as e:
        st.error(f"Не удалось загрузить tokenizer: {e}")


# Кнопка обучения / применения
if st.button("Обучение / Применить токенизатор"):
    if not st.session_state.texts:
        st.error("Нет загруженного корпуса для обучения/оценки.")
    else:
        with st.spinner("Обучаем / применяем tokenizer..."):
            # если токенизатор не загружен извне - тренируем
            if st.session_state.tokenizer_obj is None:
                tmp_dir = tempfile.mkdtemp()
                corpus_txt_path = str(Path(tmp_dir) / "corpus_for_training.txt")
                # записываем файл для тренера
                with open(corpus_txt_path, "w", encoding="utf-8") as f:
                    for t in st.session_state.texts:
                        f.write(t.replace("\n", " ") + "\n")
                try:
                    if model_choice == "BPE":
                        st.session_state.tokenizer_obj = train_bpe([corpus_txt_path], vocab_size=vocab_size, min_freq=min_freq)
                    elif model_choice == "WordPiece":
                        st.session_state.tokenizer_obj = train_wordpiece([corpus_txt_path], vocab_size=vocab_size, min_freq=min_freq)
                    else:  # Unigram
                        st.session_state.tokenizer_obj = train_unigram([corpus_txt_path], vocab_size=vocab_size)
                    st.success(f"Модель {model_choice} обучена.")
                except Exception as e:
                    st.error(f"Ошибка при обучении модели: {e}")
                    st.session_state.tokenizer_obj = None

            # если есть токенизатор - показываем статистику
            if st.session_state.tokenizer_obj is not None:
                st.subheader("Краткая информация о токенизаторе")
                try:
                    st.write("Модель:", st.session_state.tokenizer_obj.model.__class__.__name__)
                except Exception:
                    st.write("Модель загружена (тип неизвестен).")

                # вычисляем статистику
                st.session_state.stats = compute_token_statistics(st.session_state.tokenizer_obj, st.session_state.texts, top_n=50)

# Отображаем результаты, если статистика готова
if st.session_state.stats is not None:
    st.metric("Всего токенов", st.session_state.stats["total_tokens"])
    st.metric("OOV соотношение (неизвестных токенов / всего токенов)", f"{st.session_state.stats['oov_ratio']:.4f}")
    st.metric("Среднее число токенов на слово", f"{st.session_state.stats['tokens_per_word_mean']:.3f}")

    col1, col2 = st.columns(2)
    with col1:
        fig_len = px.histogram(st.session_state.stats["token_lengths"], nbins=40, title="Распределение длины токена")
        st.plotly_chart(fig_len, use_container_width=True)
    with col2:
        fig_tpw = px.histogram(st.session_state.stats["tokens_per_word_counts"], nbins=20, title="Распределение токенов по словам")
        st.plotly_chart(fig_tpw, use_container_width=True)

    st.subheader("Частовстречаемые токены")
    df_top = pd.DataFrame(st.session_state.stats["top_tokens"], columns=["Токен", "Количество"])
    st.dataframe(df_top.head(50))

    fig_top = px.bar(df_top.head(30), x="Токен", y="Количество", title="Топ 30 токенов")
    st.plotly_chart(fig_top, use_container_width=True)

    st.write(f"Медиана токенов на слово: {st.session_state.stats['tokens_per_word_median']:.3f}")

    save_col1, save_col2 = st.columns(2)
    with save_col1:
        # Сохранение tokenizer: формируем json-строку и даём кнопку скачивания (если tokenizer есть)
        try:
            tok_json_bytes = st.session_state.tokenizer_obj.to_str().encode("utf-8")
            st.download_button(
                label="Скачать tokenizer .json",
                data=tok_json_bytes,
                file_name=f"{model_choice.lower()}_v{vocab_size}.json",
                mime="application/json",
                key="download_tokenizer"
            )
        except Exception as e:
            st.error(f"Не удалось подготовить tokenizer к скачиванию: {e}")

    with save_col2:
        # Скачивание топа токенов CSV - формируем CSV и показываем кнопку
        try:
            tmpdf_bytes = df_top.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Скачать топ токенов CSV",
                data=tmpdf_bytes,
                file_name="top_tokens.csv",
                mime="text/csv",
                key="download_top_tokens"
            )
        except Exception as e:
            st.error(f"Не удалось подготовить CSV к скачиванию: {e}")

    # Сформировать и скачать HTML отчёт - генерируем HTML и предоставляем download_button сразу
    try:
        html_report = build_html_report(st.session_state.texts, st.session_state.tokenizer_obj, st.session_state.stats, title=f"Отчёт: {model_choice} словарь={vocab_size}")
        st.download_button(
            label="Сформировать и скачать HTML отчёт",
            data=html_report.encode("utf-8"),
            file_name=export_html_name,
            mime="text/html",
            key="download_html"
        )
    except Exception as e:
        st.error(f"Ошибка при формировании HTML отчёта: {e}")
