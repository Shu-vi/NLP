# Запуск: streamlit run streamlit_app.py
import streamlit as st
from gensim.models import Word2Vec, FastText, Doc2Vec
from gensim.utils import simple_preprocess
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import umap
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objs as go
import plotly.express as px

#Загрузка обученной модели
st.set_page_config(layout="wide", page_title="Исследование векторов")

st.title("Интерактивное изучение векторных представлений")

#sidebar: загрузка модели
st.sidebar.header("Выберите модель и затем загрузите обученную модель")
model_type = st.sidebar.selectbox("Тип модели", ["Word2Vec", "FastText", "Doc2Vec"])
model_file = st.sidebar.file_uploader("Загрузить обученную модель")


#инициализация/загрузка модели
model_w2v = None
model_fasttext = None
model_doc2vec = None
df_steps = None
if "df_steps" in st.session_state and st.session_state["df_steps"] is not None:
    df_steps = st.session_state["df_steps"]
df_proj = None
if "df_proj" in st.session_state and st.session_state["df_proj"] is not None:
    df_proj = st.session_state["df_proj"]
df = None
if "df" in st.session_state and st.session_state["df"] is not None:
    df = st.session_state["df"]
if model_type == "Word2Vec":
    if model_file and st.session_state.get("model_w2v") is None:
        with open("temp_model.model", "wb") as f:
            f.write(model_file.getbuffer())
        model_w2v = Word2Vec.load("temp_model.model")
        try:
            os.remove("temp_model.model")
        except OSError:
            pass
        st.session_state["model_w2v"] = model_w2v
    else:
        model_w2v = st.session_state.get("model_w2v")
elif model_type == "FastText":
    if model_file and st.session_state.get("model_fasttext") is None:
        with open("temp_model.model", "wb") as f:
            f.write(model_file.getbuffer())
        model_fasttext = FastText.load("temp_model.model")
        try:
            os.remove("temp_model.model")
        except OSError:
            pass
        st.session_state["model_fasttext"] = model_fasttext
    else:
        model_fasttext = st.session_state.get("model_fasttext")
else:#Doc2Vec
    if model_file and st.session_state.get("model_doc2vec") is None:
        with open("temp_model.model", "wb") as f:
            f.write(model_file.getbuffer())
        model_fasttext = Doc2Vec.load("temp_model.model")
        try:
            os.remove("temp_model.model")
        except OSError:
            pass
        st.session_state["model_doc2vec"] = model_doc2vec
    else:
        model_doc2vec = st.session_state.get("model_doc2vec")

#вспомогательные функции
def in_vocab(model, word):
    """
    проверка слова на наличие в словаре
    """
    if model is None:
        return False
    try:
        return word in model.wv
    except Exception:
        return False

def most_similar(model, positive=None, negative=None, topn=10):
    """
    возвращает результат из выражения вида король - мужчина + женщина (= королева)
    """
    try:
        return model.wv.most_similar(positive=positive or [], negative=negative or [], topn=topn)
    except Exception as e:
        return []

def build_html_report(title: str,
                      df_steps: pd.DataFrame | None = None,
                      df_proj: pd.DataFrame | None = None,
                      df_matrix: pd.DataFrame | None = None,
                      figs: list = None) -> str:
    """
    Формирует HTML отчёт: таблицы и графики.
    """
    figs = figs or []
    html_parts = [f"<h1>{title}</h1>",
                  "<p>Отчёт сформирован автоматически из последних доступных данных.</p>"]

    if df_steps is not None and not df_steps.empty:
        html_parts.append("<h2>Промежуточные шаги выражения</h2>")
        html_parts.append(df_steps.to_html(index=False))
    else:
        html_parts.append("<p><em>Нет данных о промежуточных шагах</em></p>")

    if df_proj is not None and not df_proj.empty:
        html_parts.append("<h2>Проекции слов на ось</h2>")
        html_parts.append(df_proj.to_html(index=True))
    else:
        html_parts.append("<p><em>Нет данных о проекциях</em></p>")

    if df_matrix is not None and not df_matrix.empty:
        html_parts.append("<h2>Матрица сходств</h2>")
        html_parts.append(df_matrix.to_html(index=True))
    else:
        html_parts.append("<p><em>Нет матрицы сходств</em></p>")

    # вставляем графики Plotly: первый с include_plotlyjs="cdn"
    for i, f in enumerate(figs):
        html_parts.append(f"<h3>График {i+1}</h3>")
        html_parts.append(f.to_html(full_html=False, include_plotlyjs=("cdn" if i == 0 else False)))

    return "\n".join(html_parts)

def cosine_between_vecs(a, b):
    """
    угол косинуса между векторами
    """
    if a is None or b is None:
        return None
    val = cosine_similarity([a], [b])[0][0]
    return float(val)

def infer_docvec(model, text):
    """
    возвращает вектор документа
    """
    if model is None:
        return None
    try:
        return model.infer_vector(simple_preprocess(text))
    except Exception:
        return None

def word_vector(model, word):
    """
    возвращает вектор слова
    """
    try:
        return model.wv[word]
    except Exception:
        return None

#UI: векторная арифметика
st.header("Интерактивная векторная арифметика")
col1, col2 = st.columns([2,1])

with col1:
    expr = st.text_input("Введите выражение (пример: сша - трамп + путин)", value="сша - трамп + путин")
    topn = st.number_input("Количество ближайших соседей (topn)", min_value=1, max_value=15, value=3)
    run_expr = st.button("Вычислить выражение")

with col2:
    st.write(f"Тип модели: {model_type}")

def parse_expression(expr_str):
    """
    парсинг выражений вида: w1 - w2 + w3 - w4
    """
    # Простая лексическая парсировка: слова и +/-
    tokens = expr_str.replace("+", " + ").replace("-", " - ").split()
    ops = []
    current = None
    # схема: первый токен может быть +/- или словом
    sign = 1
    vec_ops = []
    for t in tokens:
        if t == "+":
            sign = 1
        elif t == "-":
            sign = -1
        else:
            vec_ops.append((t, sign))
            sign = 1
    return vec_ops

def compute_intermediate_vectors(model, expr_ops):
    #статистика
    intermediate = []
    #результирующий вектор со всеми вычислениями, здесь будет храниться вычисления вида сша-трамп+путин
    result = np.zeros(model.wv.vector_size)
    for word, sign in expr_ops:
        if not in_vocab(model, word):
            intermediate.append({"word": word, "present": False, "vec": None, "result_after": None})
            continue
        vec = word_vector(model, word) * sign
        result = result + vec
        intermediate.append({"word": word, "present": True, "vec": vec.copy(), "result_after": result.copy()})
    return intermediate, result

#подсчёт векторной арифметики
if run_expr:
    #выбрать активную модель
    active_model = model_w2v if model_type=="Word2Vec" else (model_fasttext if model_type=="FastText" else model_doc2vec)
    if active_model is None:
        st.error("Модель не загружена")
    else:
        ops = parse_expression(expr)
        intermediate, final_vec = compute_intermediate_vectors(active_model, ops)

        # показываем таблицу промежуточных шагов
        rows = []
        for i, s in enumerate(intermediate):
            if not s["present"]:
                rows.append({"шаг": i+1, "слово": s["word"], "в словаре": False, "наиболее похожие": None})
            else:
                ms = most_similar(active_model, positive=[s["vec"]], topn=topn)
                rows.append({
                    "шаг": i+1,
                    "слово": s["word"],
                    "в словаре": True,
                    "наиболее похожие": ", ".join([f"{w} ({float(sim):.3f})" for w, sim in ms])
                })
        df_steps = pd.DataFrame(rows)
        st.session_state["df_steps"] = df_steps
        st.subheader("Промежуточные шаги")
        st.dataframe(df_steps)

        #ближайшие соседи для финального вектора
        st.subheader("Результат выражения — ближайшие слова")
        try:
            final_neighbors = active_model.wv.similar_by_vector(final_vec, topn=topn)
        except Exception:
            final_neighbors = []
        st.write(final_neighbors)

        #визуализация финального вектора в 2D
        st.subheader("2D проекция: промежуточные и итоговый векторы")
        #соберём векторы для рисования: все оригинальные слов-векторов и результат
        vis_vectors = []
        vis_labels = []
        for s in intermediate:
            if s["present"]:
                vis_vectors.append(s["vec"])
                vis_labels.append(f"{s['word']} (шаг)")
        vis_vectors.append(final_vec)
        vis_labels.append("финальный вектор")
        vis_vectors_np = np.array(vis_vectors)
        reducer = UMAP_OR_PCA = None
        try:
            reducer = umap.UMAP(n_components=2, random_state=42)
            proj = reducer.fit_transform(vis_vectors_np)
        except Exception:
            reducer = PCA(n_components=2)
            proj = reducer.fit_transform(vis_vectors_np)
        fig = px.scatter(x=proj[:,0], y=proj[:,1], text=vis_labels, title="2D проекция")
        st.plotly_chart(fig, use_container_width=True)

#UI: косинусное расстояние и матрица сходств
st.header("Калькулятор косинусного сходства и матрица близостей")
col1, col2 = st.columns(2)
with col1:
    word_a = st.text_input("Слово A", value="путин", key="cos_a")
    word_b = st.text_input("Слово B", value="президент", key="cos_b")
    calc_cos = st.button("Посчитать косинусное сходство")
with col2:
    words_for_matrix = st.text_area("Список слов для матрицы (через запятую)", value="россия,трамп,китай,спорт")
    calc_matrix = st.button("Построить матрицу сходств")

if calc_cos:
    active_model = model_w2v if model_type=="Word2Vec" else (model_fasttext if model_type=="FastText" else model_doc2vec)
    if active_model is None:
        st.error("Модель не загружена")
    else:
        if in_vocab(active_model, word_a) and in_vocab(active_model, word_b):
            va = word_vector(active_model, word_a)
            vb = word_vector(active_model, word_b)
            cosv = cosine_between_vecs(va, vb)
            st.metric("Косинусное сходство", f"{cosv:.4f}")
        else:
            st.error("Одно из слов отсутствует в словаре модели")

if calc_matrix:
    active_model = model_w2v if model_type=="Word2Vec" else (model_fasttext if model_type=="FastText" else model_doc2vec)
    words = [w.strip() for w in words_for_matrix.split(",") if w.strip()]
    present = [w for w in words if in_vocab(active_model, w)]
    if not present:
        st.error("Нет слов из списка в словаре модели")
    else:
        mat = np.array([word_vector(active_model, w) for w in present])
        simm = cosine_similarity(mat)
        df = pd.DataFrame(simm, index=present, columns=present)
        st.session_state["df"] = df
        st.subheader("Heatmap семантической близости")
        fig = px.imshow(df.values, x=present, y=present, color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df.style.background_gradient(cmap='RdBu_r', axis=None))

#UI: семантическая ось и проекция
st.header("Семантические оси и проекция")
axis_left = st.text_input("Слово A (лево оси)", value="мужчина", key="axis_a")
axis_right = st.text_input("Слово B (право оси)", value="женщина", key="axis_b")
words_for_proj = st.text_area("Слова для проекции (через запятую)", value="король,королева,президент,работник,няня")
do_proj = st.button("Произвести проекцию на ось")

def project_on_axis(model, left, right, targets):
    axis = word_vector(model, left) - word_vector(model, right)
    scores = {}
    for w in targets:
        if in_vocab(model, w):
            vec = word_vector(model, w)
            #если score > 0 то относится к левому, иначе к правому
            score = cosine_similarity([vec], [axis])[0][0]
            scores[w] = float(score)
        else:
            scores[w] = None
    return scores, axis

if do_proj:
    active_model = model_w2v if model_type=="Word2Vec" else (model_fasttext if model_type=="FastText" else model_doc2vec)
    targets = [w.strip() for w in words_for_proj.split(",") if w.strip()]
    if not in_vocab(active_model, axis_left) or not in_vocab(active_model, axis_right):
        st.error("Одна из опорных слов отсутствует в модели")
    else:
        scores, axis_vec = project_on_axis(active_model, axis_left, axis_right, targets)
        df_proj = pd.DataFrame.from_dict(scores, orient='index', columns=['projection']).sort_values('projection', ascending=False)
        st.session_state["df_proj"] = df_proj
        st.dataframe(df_proj)
        st.subheader("График проекций")
        fig = px.bar(df_proj.reset_index().rename(columns={'index':'word'}), x='word', y='projection', color='projection', color_continuous_scale='RdBu')
        st.plotly_chart(fig, use_container_width=True)

#UI: граф семантических связей
st.header("Граф семантических связей")
graph_seed = st.text_input("Слово (центр графа)", value="россия", key="graph_seed")
graph_depth = st.slider("Глубина (уровней соседей)", 1, 3, 2)
graph_topn = st.slider("TopN соседей на уровень", 1, 8, 5)

def build_similarity_graph(model, seed, depth=2, topn=5):
    G = nx.Graph()
    visited = set()
    def expand(node, d):
        if d>depth:
            return
        visited.add(node)
        if not in_vocab(model, node):
            return
        try:
            neighbors = model.wv.most_similar(node, topn=topn)
        except Exception:
            neighbors = []
        for nb, sim in neighbors:
            G.add_node(node)
            G.add_node(nb)
            G.add_edge(node, nb, weight=float(sim))
            if nb not in visited:
                expand(nb, d+1)
    expand(seed, 1)
    return G

if st.button("Построить граф"):
    active_model = model_w2v if model_type=="Word2Vec" else (model_fasttext if model_type=="FastText" else model_doc2vec)
    if not in_vocab(active_model, graph_seed):
        st.error("Корневое слово отсутствует в модели")
    else:
        G = build_similarity_graph(active_model, graph_seed, depth=graph_depth, topn=graph_topn)
        st.write(f"Узлы: {len(G.nodes())}, Рёбра: {len(G.edges())}")
        #визуализация через plotly
        pos = nx.spring_layout(G, seed=42)
        edge_x = []
        edge_y = []
        for e in G.edges():
            x0, y0 = pos[e[0]]
            x1, y1 = pos[e[1]]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
        node_x = []
        node_y = []
        texts = []
        for n in G.nodes():
            x, y = pos[n]
            node_x.append(x)
            node_y.append(y)
            texts.append(n)
        edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=0.5, color='#888'), hoverinfo='none')
        node_trace = go.Scatter(
            x=node_x, y=node_y, mode='markers+text', text=texts, textposition="top center",
            hoverinfo='text', marker=dict(showscale=False, size=10, color='skyblue', line_width=2)
        )
        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(showlegend=False, margin=dict(b=20,l=5,r=5,t=40))
        st.plotly_chart(fig, use_container_width=True)


#UI: генерация отчёта
st.header("Генерация отчёта")
report_title = st.text_input("Заголовок отчёта", value="Отчёт")
report_btn = st.button("Сгенерировать отчёт")


if report_btn:
    try:
        last_steps = df_steps
    except Exception:
        last_steps = pd.DataFrame()
    try:
        last_proj = df_proj
    except Exception:
        last_proj = pd.DataFrame()
    try:
        last_mat = df
    except Exception:
        last_mat = pd.DataFrame()

    # добавляем последние графики, если есть
    figs_to_add = []
    if "fig" in globals() and fig is not None:
        figs_to_add.append(fig)

    html_report = build_html_report(report_title, last_steps, last_proj, last_mat, figs_to_add)

    st.download_button(
        label="Скачать HTML отчёт",
        data=html_report.encode("utf-8"),
        file_name="report.html",
        mime="text/html",
    )


st.sidebar.header("Для doc2vec только схожести предложений")