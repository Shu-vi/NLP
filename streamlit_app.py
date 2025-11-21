import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, precision_score, recall_score, f1_score
import json

# Импортируем ваши модули
try:
    from use_ml import predict_sentiment, predict_category, predict_categorys
except ImportError:
    st.error("Модуль use_ml не найден")
try:
    from use_nn import predict_sentiment as nn_predict_sentiment
    from use_nn import predict_category as nn_predict_category
    from use_nn import predict_categorys as nn_predict_categorys
except ImportError:
    st.error("Модуль use_nn не найден")
try:
    from use_transformer import predict_sentiment as tf_predict_sentiment
    from use_transformer import predict_category as tf_predict_category
    from use_transformer import predict_categorys as tf_predict_categorys
except ImportError:
    st.error("Модуль use_transformer не найден")

# Настройка страницы
st.set_page_config(
    page_title="Анализ классификаторов текста",
    page_icon="📊",
    layout="wide"
)


def load_models(task_type):
    """Загрузка моделей в зависимости от типа задачи"""
    models = {}

    if task_type == "Бинарная":
        try:
            models["Классическая ML"] = predict_sentiment()
        except:
            pass
        try:
            models["Нейросеть"] = nn_predict_sentiment()
        except:
            pass
        try:
            models["Трансформер"] = tf_predict_sentiment()
        except:
            pass

    elif task_type == "Многоклассовая":
        try:
            models["Классическая ML"] = predict_category()
        except:
            pass
        try:
            models["Нейросеть"] = nn_predict_category()
        except:
            pass
        try:
            models["Трансформер"] = tf_predict_category()
        except:
            pass

    elif task_type == "Многометочная":
        try:
            models["Классическая ML"] = predict_categorys()
        except:
            pass
        try:
            models["Нейросеть"] = nn_predict_categorys()
        except:
            pass
        try:
            models["Трансформер"] = tf_predict_categorys()
        except:
            pass

    return models


def plot_probabilities(probs, labels, model_name):
    """Визуализация вероятностей"""
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(labels))

    if isinstance(probs, (np.ndarray, list)) and len(probs) > 1:
        # Многоклассовая или многометочная
        ax.barh(y_pos, probs, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel('Вероятность')
        ax.set_title(f'Вероятности классов - {model_name}')
    else:
        # Бинарная
        binary_probs = [1 - probs, probs] if isinstance(probs, (int, float)) else [1 - probs[0], probs[0]]
        binary_labels = ['Negative', 'Positive']
        ax.barh([0, 1], binary_probs, align='center')
        ax.set_yticks([0, 1])
        ax.set_yticklabels(binary_labels)
        ax.set_xlabel('Вероятность')
        ax.set_title(f'Вероятности классов - {model_name}')

    plt.tight_layout()
    return fig


def calculate_and_display_binary_metrics(true_labels, predictions):
    """Расчет и отображение метрик для бинарной классификации"""
    # Преобразуем true_labels в числовой формат
    y_true = [1 if label == 'positive' else 0 for label in true_labels]
    y_pred = [1 if pred['probs'] >= 0.5 else 0 for pred in predictions]
    y_scores = [pred['probs'] for pred in predictions]

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_scores)

    # Визуализация
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # ROC curve
    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend(loc="lower right")

    # Precision-Recall curve
    ax2.plot(recall, precision, color='blue', lw=2)
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')

    st.pyplot(fig)

    # Матрица ошибок
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

    # Отчет классификации
    st.subheader("Отчет классификации")
    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df, use_container_width=True)


def calculate_and_display_multiclass_metrics(true_labels, predictions):
    """Расчет и отображение метрик для многоклассовой классификации"""
    # Получаем все уникальные классы
    all_classes = list(set(true_labels))

    # Предсказанные классы (класс с максимальной вероятностью)
    y_pred = [pred['labels'][np.argmax(pred['probs'])] for pred in predictions]
    y_true = true_labels

    # Матрица ошибок
    cm = confusion_matrix(y_true, y_pred, labels=all_classes)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=all_classes, yticklabels=all_classes)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    st.pyplot(fig)

    # Отчет классификации
    st.subheader("Отчет классификации")
    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df, use_container_width=True)

    # Визуализация точности по классам
    class_report = classification_report(y_true, y_pred, output_dict=True)
    classes_metrics = {}
    for class_name in all_classes:
        if class_name in class_report:
            classes_metrics[class_name] = {
                'Precision': class_report[class_name]['precision'],
                'Recall': class_report[class_name]['recall'],
                'F1-Score': class_report[class_name]['f1-score']
            }

    metrics_df = pd.DataFrame(classes_metrics).T
    fig, ax = plt.subplots(figsize=(12, 6))
    metrics_df.plot(kind='bar', ax=ax)
    ax.set_title('Метрики по классам')
    ax.set_ylabel('Score')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    st.pyplot(fig)


def calculate_and_display_multilabel_metrics(true_labels, predictions):
    """Расчет и отображение метрик для многометочной классификации"""
    # Получаем все возможные метки из предсказаний
    all_labels = predictions[0]['labels']

    # Создаем бинарные матрицы для истинных и предсказанных меток
    y_true_binary = np.zeros((len(true_labels), len(all_labels)))
    y_pred_binary = np.zeros((len(predictions), len(all_labels)))

    for i, (true_label_list, pred) in enumerate(zip(true_labels, predictions)):
        for j, label in enumerate(all_labels):
            # Истинные метки
            if label in true_label_list:
                y_true_binary[i, j] = 1

            # Предсказанные метки (порог 0.5)
            if pred['probs'][j] >= 0.5:
                y_pred_binary[i, j] = 1

    # Вычисляем метрики для каждой метки
    metrics_per_label = {}
    for j, label in enumerate(all_labels):
        metrics_per_label[label] = {
            'Precision': precision_score(y_true_binary[:, j], y_pred_binary[:, j]),
            'Recall': recall_score(y_true_binary[:, j], y_pred_binary[:, j]),
            'F1-Score': f1_score(y_true_binary[:, j], y_pred_binary[:, j]),
            'Support': np.sum(y_true_binary[:, j])
        }

    # Сводная таблица метрик
    st.subheader("Метрики по меткам")
    metrics_df = pd.DataFrame(metrics_per_label).T
    st.dataframe(metrics_df, use_container_width=True)

    # Визуализация метрик
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Precision по меткам
    axes[0, 0].barh(range(len(all_labels)), [metrics_per_label[label]['Precision'] for label in all_labels])
    axes[0, 0].set_yticks(range(len(all_labels)))
    axes[0, 0].set_yticklabels(all_labels)
    axes[0, 0].set_title('Precision по меткам')
    axes[0, 0].set_xlim(0, 1)

    # Recall по меткам
    axes[0, 1].barh(range(len(all_labels)), [metrics_per_label[label]['Recall'] for label in all_labels])
    axes[0, 1].set_yticks(range(len(all_labels)))
    axes[0, 1].set_yticklabels(all_labels)
    axes[0, 1].set_title('Recall по меткам')
    axes[0, 1].set_xlim(0, 1)

    # F1-Score по меткам
    axes[1, 0].barh(range(len(all_labels)), [metrics_per_label[label]['F1-Score'] for label in all_labels])
    axes[1, 0].set_yticks(range(len(all_labels)))
    axes[1, 0].set_yticklabels(all_labels)
    axes[1, 0].set_title('F1-Score по меткам')
    axes[1, 0].set_xlim(0, 1)

    # Support по меткам
    axes[1, 1].barh(range(len(all_labels)), [metrics_per_label[label]['Support'] for label in all_labels])
    axes[1, 1].set_yticks(range(len(all_labels)))
    axes[1, 1].set_yticklabels(all_labels)
    axes[1, 1].set_title('Support (количество примеров) по меткам')

    plt.tight_layout()
    st.pyplot(fig)

    # Примеры предсказаний
    st.subheader("Примеры предсказаний")
    sample_indices = np.random.choice(len(predictions), min(5, len(predictions)), replace=False)

    for idx in sample_indices:
        with st.expander(f"Пример {idx + 1}"):
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Истинные метки:**")
                st.write(true_labels[idx])

            with col2:
                st.write("**Предсказанные метки:**")
                predicted_labels = [all_labels[i] for i, prob in enumerate(predictions[idx]['probs']) if prob >= 0.5]
                st.write(predicted_labels)

                st.write("**Вероятности:**")
                prob_df = pd.DataFrame({
                    'Метка': all_labels,
                    'Вероятность': predictions[idx]['probs']
                }).sort_values('Вероятность', ascending=False)
                st.dataframe(prob_df, use_container_width=True)


def process_test_file(uploaded_file, task_type):
    """Обработка загруженного JSONL файла"""
    data = []
    for line in uploaded_file:
        data.append(json.loads(line.decode('utf-8')))

    df = pd.DataFrame(data)
    return df


def calculate_metrics(df, predictions, task_type):
    """Расчет метрик качества"""
    if task_type == "Бинарная":
        y_true = df['label'].apply(lambda x: 1 if x == 'positive' else 0)
        y_pred = [1 if pred['probs'] >= 0.5 else 0 for pred in predictions]
        y_scores = [pred['probs'] for pred in predictions]

        # ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_scores)

        return {
            'fpr': fpr,
            'tpr': tpr,
            'roc_auc': roc_auc,
            'precision': precision,
            'recall': recall,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_scores': y_scores
        }

    elif task_type == "Многоклассовая":
        # Для многоклассовой нужна более сложная обработка
        return {"message": "Многоклассовые метрики требуют дополнительной реализации"}

    else:
        return {"message": "Многометочные метрики требуют дополнительной реализации"}


# Основной интерфейс
st.title("📊 Анализ классификаторов текста")

# Сайдбар для навигации
st.sidebar.title("Навигация")
app_mode = st.sidebar.selectbox(
    "Выберите режим",
    ["Интерактивная классификация", "Анализ тестовой выборки"],
    key="main_navigation"  # Уникальный ключ
)

# Интерактивная классификация
if app_mode == "Интерактивная классификация":
    st.header("🔍 Интерактивная классификация")

    col1, col2 = st.columns([1, 1])

    with col1:
        task_type = st.selectbox(
            "Тип задачи",
            ["Бинарная", "Многоклассовая", "Многометочная"],
            key="interactive_task_type"  # Уникальный ключ
        )

        available_models = list(load_models(task_type).keys())
        if not available_models:
            st.error("Нет доступных моделей для выбранного типа задачи")
            st.stop()

        selected_models = st.multiselect(
            "Выберите модели для сравнения",
            available_models,
            default=available_models[0] if available_models else None,
            key="interactive_models"  # Уникальный ключ
        )

        text_input = st.text_area(
            "Введите текст для классификации",
            height=150,
            placeholder="Введите текст здесь...",
            key="interactive_text_input"  # Уникальный ключ
        )

    with col2:
        if text_input and selected_models:
            models = load_models(task_type)

            for model_name in selected_models:
                st.subheader(f"Модель: {model_name}")

                try:
                    result = models[model_name](text_input)

                    # Отображение результатов
                    if task_type == "Бинарная":
                        sentiment = "Positive" if result['probs'] >= 0.5 else "Negative"
                        confidence = result['probs'] if result['probs'] >= 0.5 else 1 - result['probs']

                        st.write(f"**Результат**: {sentiment}")
                        st.write(f"**Уверенность**: {confidence:.3f}")

                        # Визуализация вероятностей
                        fig = plot_probabilities(result['probs'], result.get('labels', ['Negative', 'Positive']),
                                                 model_name)
                        st.pyplot(fig)

                    else:
                        if task_type == "Многоклассовая":
                            predicted_idx = np.argmax(result['probs'])
                            predicted_label = result['labels'][predicted_idx]
                            confidence = result['probs'][predicted_idx]

                            st.write(f"**Предсказанный класс**: {predicted_label}")
                            st.write(f"**Уверенность**: {confidence:.3f}")

                        else:  # Многометочная
                            predicted_labels = [result['labels'][i] for i, prob in enumerate(result['probs']) if
                                                prob >= 0.5]
                            st.write(f"**Предсказанные классы**: {', '.join(predicted_labels)}")

                        # Визуализация вероятностей
                        fig = plot_probabilities(result['probs'], result['labels'], model_name)
                        st.pyplot(fig)

                        # Таблица вероятностей
                        prob_df = pd.DataFrame({
                            'Класс': result['labels'],
                            'Вероятность': result['probs']
                        }).sort_values('Вероятность', ascending=False)

                        st.dataframe(prob_df, use_container_width=True)

                except Exception as e:
                    st.write("График не поддерживается у данной модели")

# Анализ тестовой выборки
elif app_mode == "Анализ тестовой выборки":
    st.header("📈 Анализ тестовой выборки")

    uploaded_file = st.file_uploader(
        "Загрузите JSONL файл с тестовой выборкой",
        type=['jsonl'],
        help="Файл должен содержать поля 'text' и 'label' (для бинарной/многоклассовой) или 'labels' (для многометочной)",
        key="file_uploader"
    )

    if uploaded_file:
        task_type = st.selectbox(
            "Тип задачи для анализа",
            ["Бинарная", "Многоклассовая", "Многометочная"],
            key="analysis_task_type"
        )

        available_models = list(load_models(task_type).keys())
        if not available_models:
            st.error("Нет доступных моделей для выбранного типа задачи")
            st.stop()

        selected_model = st.selectbox(
            "Выберите модель для анализа",
            available_models,
            key="analysis_model"
        )

        if st.button("Запустить анализ", key="analyze_button"):
            with st.spinner("Обработка данных..."):
                # Загрузка и обработка данных
                df = process_test_file(uploaded_file, task_type)
                st.write(f"Загружено {len(df)} примеров")

                # Проверка структуры данных
                st.subheader("Структура данных")
                st.dataframe(df.head(), use_container_width=True)

                if task_type == "Многометочная" and 'labels' not in df.columns:
                    st.error("Для многометочной классификации в файле должно быть поле 'labels'")
                    st.stop()
                elif task_type != "Многометочная" and 'label' not in df.columns:
                    st.error("Для бинарной и многоклассовой классификации в файле должно быть поле 'label'")
                    st.stop()

                # Предсказания
                model = load_models(task_type)[selected_model]
                predictions = []
                true_labels = []

                progress_bar = st.progress(0)
                for i, row in df.iterrows():
                    try:
                        result = model(row['text'])
                        predictions.append(result)

                        # Сохраняем истинные метки в нужном формате
                        if task_type == "Многометочная":
                            true_labels.append(row['labels'])
                        else:
                            true_labels.append(row['label'])

                    except Exception as e:
                        st.error(f"Ошибка при обработке примера {i}: {str(e)}")
                        predictions.append(None)
                        true_labels.append(None)

                    progress_bar.progress((i + 1) / len(df))

                # Удаляем примеры с ошибками
                valid_indices = [i for i, pred in enumerate(predictions) if pred is not None]
                predictions = [predictions[i] for i in valid_indices]
                true_labels = [true_labels[i] for i in valid_indices]

                st.write(f"Успешно обработано {len(predictions)} из {len(df)} примеров")

                # Расчет и отображение метрик для разных типов задач
                if task_type == "Бинарная":
                    calculate_and_display_binary_metrics(true_labels, predictions)

                elif task_type == "Многоклассовая":
                    calculate_and_display_multiclass_metrics(true_labels, predictions)

                elif task_type == "Многометочная":
                    calculate_and_display_multilabel_metrics(true_labels, predictions)


# Информация в сайдбаре
st.sidebar.markdown("---")
st.sidebar.info("""
**Инструкция:**
1. **Интерактивная классификация**: Тестируйте модели на произвольном тексте
2. **Анализ тестовой выборки**: Загрузите JSONL файл для оценки качества
""")

# CSS для улучшения внешнего вида
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)