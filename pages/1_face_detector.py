import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
import base64
import time
import pandas as pd
from pathlib import Path
import sys
import os

# ========== БЕЗОПАСНЫЙ ИМПОРТ OPENCV ==========
try:
    import cv2
except ImportError:
    import subprocess
    with st.spinner("📦 Установка OpenCV для облачной среды..."):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python-headless"])
    import cv2

# ========== НАСТРОЙКИ СТРАНИЦЫ ==========
st.set_page_config(
    page_title="Face Detection App",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== ЗАГРУЗКА МОДЕЛИ ==========
@st.cache_resource
def load_model(model_path):
    """Загрузка обученной YOLO модели"""
    try:
        if not Path(model_path).exists():
            st.error(f"❌ Модель не найдена по пути: {model_path}")
            return None
        
        model = YOLO(model_path)
        
        # Определяем устройство
        device = 'cuda' if model.model is not None and hasattr(model.model, 'device') else 'cpu'
        st.sidebar.success(f"✅ Модель загружена | Устройство: {device.upper()}")
        
        return model
    except Exception as e:
        st.error(f"❌ Ошибка загрузки модели: {e}")
        return None

# ========== ФУНКЦИИ ДЛЯ ОБРАБОТКИ ==========
def process_image(model, image, conf_threshold=0.5, iou_threshold=0.45):
    """Обработка изображения и детекция лиц"""
    results = model(image, conf=conf_threshold, iou=iou_threshold)
    return results

def draw_detections(image, results, conf_threshold=0.5):
    """Отрисовка результатов детекции"""
    img_copy = image.copy()
    
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        names = results[0].names
        
        for i, box in enumerate(boxes):
            # Получаем координаты
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            confidence = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())
            label = names[class_id]
            
            # Пропускаем если уверенность ниже порога
            if confidence < conf_threshold:
                continue
            
            # Цвет в зависимости от уверенности (от желтого к зеленому)
            color = (0, int(255 * confidence), int(255 * (1 - confidence)))
            
            # Рисуем bounding box
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
            
            # Добавляем текст с уверенностью
            text = f"{label}: {confidence:.2%}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(img_copy, (x1, y1 - text_size[1] - 5), 
                         (x1 + text_size[0], y1), color, -1)
            cv2.putText(img_copy, text, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return img_copy

def get_statistics(results, conf_threshold=0.5):
    """Получение статистики детекции"""
    if len(results) == 0 or results[0].boxes is None:
        return {
            'total_faces': 0,
            'avg_confidence': 0,
            'max_confidence': 0,
            'min_confidence': 0,
            'face_sizes': [],
            'confidences': [],
            'boxes': []
        }
    
    boxes = results[0].boxes
    confidences = boxes.conf.cpu().numpy()
    
    # Фильтруем по порогу
    mask = confidences >= conf_threshold
    confidences = confidences[mask]
    boxes_xyxy = boxes.xyxy.cpu().numpy()[mask]
    
    # Вычисляем размеры лиц (площадь bounding box)
    face_sizes = []
    for box in boxes_xyxy:
        width = box[2] - box[0]
        height = box[3] - box[1]
        face_sizes.append(width * height)
    
    return {
        'total_faces': len(confidences),
        'avg_confidence': np.mean(confidences) if len(confidences) > 0 else 0,
        'max_confidence': np.max(confidences) if len(confidences) > 0 else 0,
        'min_confidence': np.min(confidences) if len(confidences) > 0 else 0,
        'face_sizes': face_sizes,
        'confidences': confidences,
        'boxes': boxes_xyxy
    }

def create_stats_plots(statistics):
    """Создание графиков статистики"""
    if statistics['total_faces'] == 0:
        return None, None
    
    # График 1: Распределение уверенности
    fig_conf = go.Figure()
    fig_conf.add_trace(go.Histogram(
        x=statistics['confidences'],
        name='Уверенность',
        marker_color='green',
        opacity=0.7,
        nbinsx=20
    ))
    fig_conf.update_layout(
        title="Распределение уверенности детекции",
        xaxis_title="Уверенность",
        yaxis_title="Количество",
        height=350,
        showlegend=False
    )
    
    # График 2: Размеры лиц
    fig_sizes = go.Figure()
    fig_sizes.add_trace(go.Histogram(
        x=statistics['face_sizes'],
        name='Размеры лиц',
        marker_color='blue',
        opacity=0.7,
        nbinsx=20
    ))
    fig_sizes.update_layout(
        title="Распределение размеров лиц",
        xaxis_title="Размер (пиксели²)",
        yaxis_title="Количество",
        height=350,
        showlegend=False
    )
    
    return fig_conf, fig_sizes

def create_confidence_gauge(avg_confidence):
    """Создание индикатора средней уверенности"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = avg_confidence * 100,
        title = {'text': "Средняя уверенность (%)"},
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkgreen"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 75], 'color': "gray"},
                {'range': [75, 100], 'color': "darkgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

def download_image(image, filename="detected_faces.jpg"):
    """Создание ссылки для скачивания изображения"""
    buffered = BytesIO()
    if isinstance(image, np.ndarray):
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        image_pil = image
    
    image_pil.save(buffered, format="JPEG", quality=95)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/jpg;base64,{img_str}" download="{filename}">📥 Скачать результат</a>'
    return href
def get_example_image(type="group"):
    """Получение примера изображения из интернета"""
    import requests
    
    # Словарь с URL реальных изображений
    examples = {
        "group": "https://shkolamoskva.ru/wp-content/uploads/2023/07/photo_5853940747303235966_y.jpg",  # Группа людей
        "portrait": "https://img.freepik.com/free-photo/smiling-portrait-studio-woman_1303-2289.jpg",  # Портрет
        "crowd": "https://images.theconversation.com/files/638065/original/file-20241212-15-93ub7o.jpg",  # Толпа
        "family": "https://i.pinimg.com/originals/8c/36/f4/8c36f41aaf025ce904710814284db9d3.jpg"  # Семья
    }
    
    try:
        url = examples.get(type, examples["group"])
        response = requests.get(url, timeout=10)
        return BytesIO(response.content)
    except Exception as e:
        st.warning(f"Не удалось загрузить пример: {e}")
        # Возвращаем заглушку
        img = np.ones((400, 400, 3), dtype=np.uint8) * 128
        _, buffer = cv2.imencode('.jpg', img)
        return BytesIO(buffer.tobytes())    

# ========== ОСНОВНОЙ ИНТЕРФЕЙС ==========
def main():
    st.title("👤 Face Detection System")
    st.markdown("*Обнаружение лиц с помощью обученной YOLO модели*")
    
    # Боковая панель с настройками
    with st.sidebar:
        st.header("⚙️ Настройки")
        
        # Путь к модели
        default_model_path = "best.pt"
        model_path = st.text_input(
            "Путь к модели",
            value=default_model_path,
            help="Укажите путь к файлу .pt с обученной моделью"
        )
        
        st.divider()
        
        # Параметры детекции
        st.subheader("🎯 Параметры детекции")
        conf_threshold = st.slider(
            "Порог уверенности",
            min_value=0.1,
            max_value=0.95,
            value=0.5,
            step=0.05,
            help="Минимальный порог уверенности для детекции лица"
        )
        
        iou_threshold = st.slider(
            "Порог IoU (NMS)",
            min_value=0.1,
            max_value=0.9,
            value=0.45,
            step=0.05,
            help="Порог для подавления пересекающихся框"
        )
        
        st.divider()
        
        # Визуальные настройки
        st.subheader("🎨 Визуализация")
        show_labels = st.checkbox("Показывать подписи", value=True)
        show_confidence = st.checkbox("Показывать уверенность", value=True)
        
        st.divider()
        
        # Информация о модели
        st.subheader("ℹ️ О модели")
        st.info(
            f"**Модель:** YOLO26n\n\n"
            f"**Тип:** Детекция лиц\n\n"
            f"**Эпохи:** 20\n\n"
            f"**Размер:** 256x256\n\n"
            f"**mAP50:** 0.778\n\n"
            f"**mAP50-95:** 0.495"
        )
    
    # Основная область
    col1, col2 = st.columns([1, 1])
    
    # Левая колонка - загрузка изображения
    with col1:
        st.subheader("📤 Загрузка изображения")
        
        # Выбор источника
        source = st.radio(
            "Выберите источник:",
            ["📁 Загрузить файл", "🌐 Использовать пример", "📸 Сделать снимок"],
            horizontal=True
        )
        
        image = None
        
        if source == "📁 Загрузить файл":
            uploaded_file = st.file_uploader(
                "Выберите изображение",
                type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
                help="Поддерживаются форматы: JPG, PNG, BMP, WEBP"
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Исходное изображение", use_container_width=True)
        
        elif source == "🌐 Использовать пример":
            col_ex1, col_ex2 = st.columns(2)

            with col_ex1:
                if st.button("👥 Группа людей", use_container_width=True):
                    with st.spinner("🖼️ Загрузка..."):
                        image_data = get_example_image("group")
                        st.session_state.current_image = Image.open(image_data)
                        st.rerun()  # Принудительно перерисовываем
            with col_ex2:
                if st.button("👤 Портрет", use_container_width=True):
                    with st.spinner("🖼️ Загрузка..."):
                        image_data = get_example_image("portrait")
                        st.session_state.current_image = Image.open(image_data)
                        st.rerun()

                        # Отображаем текущее изображение из session_state
            if "current_image" in st.session_state and st.session_state.current_image is not None:
                image = st.session_state.current_image
                st.image(image, caption="Выбран пример", use_container_width=True)
            else:
                image = None            
        
        else:  # Сделать снимок
            camera_image = st.camera_input("Сделайте снимок")
            if camera_image:
                image = Image.open(camera_image)
                st.image(image, caption="Сделан снимок", use_container_width=True)
        
        # Кнопка обработки
        st.divider()
        process_button = st.button(
            "🔍 ОБНАРУЖИТЬ ЛИЦА",
            type="primary",
            use_container_width=True,
            disabled=image is None
        )
    
    # Правая колонка - результаты
    with col2:
        st.subheader("🎯 Результаты детекции")
        
        if process_button and image is not None:
            # Загружаем модель
            with st.spinner("🔄 Загрузка модели..."):
                model = load_model(model_path)
            
            if model is not None:
                # Обрабатываем изображение
                with st.spinner("🔍 Анализ изображения..."):
                    start_time = time.time()
                    
                    # Конвертируем PIL в numpy
                    image_np = np.array(image)
                    if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                    
                    # Детекция
                    results = process_image(model, image_np, conf_threshold, iou_threshold)
                    
                    inference_time = time.time() - start_time
                    
                    # Получаем статистику
                    stats = get_statistics(results, conf_threshold)
                    
                    # Отрисовываем результат
                    result_image = draw_detections(image_np, results, conf_threshold)
                    result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                    
                    # Показываем результат
                    st.image(result_image_rgb, caption="Результат детекции", use_container_width=True)
                    
                    # Статистика
                    col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
                    
                    with col_metric1:
                        st.metric("👤 Обнаружено лиц", stats['total_faces'])
                    with col_metric2:
                        st.metric("📊 Средняя уверенность", f"{stats['avg_confidence']:.1%}")
                    with col_metric3:
                        st.metric("🎯 Макс. уверенность", f"{stats['max_confidence']:.1%}")
                    with col_metric4:
                        st.metric("⚡ Время обработки", f"{inference_time:.2f} сек")
                    
                    # Кнопка скачивания
                    st.markdown(download_image(result_image_rgb), unsafe_allow_html=True)
                    
                    # Детальная статистика
                    if stats['total_faces'] > 0:
                        with st.expander("📊 Детальная статистика и визуализация", expanded=False):
                            # Графики
                            fig_conf, fig_sizes = create_stats_plots(stats)
                            if fig_conf and fig_sizes:
                                col_g1, col_g2 = st.columns(2)
                                with col_g1:
                                    st.plotly_chart(fig_conf, use_container_width=True)
                                with col_g2:
                                    st.plotly_chart(fig_sizes, use_container_width=True)
                            
                            # Индикатор уверенности
                            if stats['avg_confidence'] > 0:
                                st.plotly_chart(create_confidence_gauge(stats['avg_confidence']), 
                                               use_container_width=True)
                            
                            # Таблица с деталями
                            st.write("**Детали по каждому лицу:**")
                            details_df = pd.DataFrame({
                                'Лицо #': range(1, stats['total_faces'] + 1),
                                'Уверенность': [f"{c:.1%}" for c in stats['confidences']],
                                'Размер (px²)': [f"{int(s):,}" for s in stats['face_sizes']],
                                'Позиция (x, y)': [f"({int(box[0])}, {int(box[1])})" for box in stats['boxes']]
                            })
                            st.dataframe(details_df, use_container_width=True, hide_index=True)
                    else:
                        st.warning("⚠️ Лица не обнаружены. Попробуйте уменьшить порог уверенности.")
            
            else:
                st.error("❌ Не удалось загрузить модель. Проверьте путь к файлу.")
        
        else:
            # Плейсхолдер
            placeholder = np.ones((400, 400, 3), dtype=np.uint8) * 240
            st.image(placeholder, caption="Здесь появится результат", use_container_width=True)
            st.caption("👈 Загрузите изображение и нажмите кнопку")
    
    # Дополнительная информация внизу
    st.divider()
    
    # Информация о модели в реальном времени
    with st.expander("ℹ️ Информация о модели и использовании", expanded=False):
        col_info1, col_info2 = st.columns(2)
        
        with col_info1:
            st.markdown("""
            ### 🧠 О модели
            
            **YOLO26n** - обученная модель для детекции лиц
            
            #### 📊 Метрики:
            - **mAP50:** 0.778
            - **mAP50-95:** 0.495
            - **Precision:** 0.884
            - **Recall:** 0.694
            
            #### 💾 Размер модели:
            - **Веса:** ~5.4 MB
            - **Параметры:** 2.5M
            """)
        
        with col_info2:
            st.markdown("""
            ### 📖 Руководство
            
            1. **Загрузите изображение** через файл, пример или камеру
            2. **Настройте порог** уверенности в боковой панели
            3. **Нажмите кнопку** "Обнаружить лица"
            4. **Анализируйте результаты** и статистику
            5. **Скачайте результат** при необходимости
            
            #### 💡 Советы:
            - Для лучших результатов используйте четкие изображения
            - Экспериментируйте с порогом уверенности
            - Поддерживаются групповые фото
            """)



if __name__ == "__main__":
    main()