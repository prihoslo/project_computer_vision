import streamlit as st
import folium
from streamlit_folium import st_folium
import numpy as np
from PIL import Image
import onnxruntime as ort
import torchvision.transforms as T
import math
import requests
from io import BytesIO
import time
import os
import sys

# Отключаем GPU (на бесплатном тарифе только CPU)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Проверка на запуск в Streamlit Cloud
IS_CLOUD = os.environ.get('STREAMLIT_CLOUD', False)
# ========== НАСТРОЙКИ СТРАНИЦЫ ==========
st.set_page_config(
    page_title="Forest Segmentation",
    page_icon="🌲",
    layout="wide"
)

# ========== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ==========

def lat_lon_to_tile(lat, lon, zoom):
    """Конвертация координат в тайлы"""
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile

def get_satellite_image(lat, lon, zoom=16, size=512):
    """Загрузка спутникового снимка"""
    try:
        xtile, ytile = lat_lon_to_tile(lat, lon, zoom)
        url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{ytile}/{xtile}"
        
        response = requests.get(url, headers={'User-Agent': 'ForestApp/1.0'}, timeout=10)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            return img.resize((size, size), Image.LANCZOS)
    except Exception as e:
        st.error(f"Ошибка загрузки снимка: {e}")
    
    return None

def preprocess_image(image, target_size=(512, 512)):
    """Подготовка изображения для модели"""
    transform = T.Compose([
        T.Resize(target_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(image).unsqueeze(0)
    return tensor.numpy()

def postprocess_mask(output, original_size=None, threshold=0.5):
    """Постобработка маски с настраиваемым порогом"""
    if isinstance(output, list):
        output = output[0]
    
    while output.ndim > 2:
        if output.shape[0] == 1:
            output = output[0]
        else:
            break
    
    mask = (1 / (1 + np.exp(-output))) > threshold
    mask = (mask * 255).astype(np.uint8)
    
    mask_pil = Image.fromarray(mask)
    if original_size:
        mask_pil = mask_pil.resize(original_size, Image.NEAREST)
    
    return mask_pil

def calculate_forest_percentage(mask):
    """Процент леса на изображении"""
    mask_np = np.array(mask)
    forest_pixels = np.sum(mask_np > 0)
    total_pixels = mask_np.size
    return (forest_pixels / total_pixels) * 100

def create_overlay(image, mask, alpha=0.4):
    """Наложение маски на изображение"""
    image_np = np.array(image).copy()
    mask_np = np.array(mask)
    
    overlay_color = np.array([0, 255, 0], dtype=np.uint8)
    
    result = image_np.copy()
    result[mask_np > 0] = (image_np[mask_np > 0] * (1 - alpha) + overlay_color * alpha).astype(np.uint8)
    
    return Image.fromarray(result)

# ========== ЗАГРУЗКА МОДЕЛИ ==========
@st.cache_resource
def load_model(model_path="forest_unet.onnx"):
    """Загрузка ONNX модели с Google Drive"""
    import gdown
    
    
    data_file = "forest_unet.onnx.data"
    
    # Скачиваем .data файл если его нет
    if not os.path.exists(data_file):
        with st.spinner("📥 Скачивание весов модели (~93 MB)..."):
            file_id = "1WdUTKf8tmn3HhlG0eVTr9XaT1tWJPHR-"
            url = f"https://drive.google.com/uc?id={file_id}"
            
            # В Streamlit Cloud используем /tmp для временных файлов
            if IS_CLOUD:
                data_file = f"/tmp/{data_file}"
            
            gdown.download(url, data_file, quiet=False)
            
            # Если модель ищет .data в той же папке, копируем .onnx в /tmp
            if IS_CLOUD:
                import shutil
                shutil.copy(model_path, f"/tmp/{model_path}")
                model_path = f"/tmp/{model_path}"
    
    try:
        # В Streamlit Cloud используем только CPU
        providers = ['CPUExecutionProvider']
        
        ort_session = ort.InferenceSession(
            model_path,
            providers=providers
        )
        return ort_session
    except Exception as e:
        st.warning(f"⚠️ Модель не загружена: {e}")
        return None
# ========== СОЗДАНИЕ КАРТЫ ==========
def create_map(lat, lon, zoom):
    """Создание простой карты"""
    m = folium.Map(
        location=[lat, lon],
        zoom_start=zoom,
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='ESRI',
        control_scale=True
    )
    return m

# ========== ФУНКЦИЯ ПЕРЕСЧЕТА МАСКИ ПРИ ИЗМЕНЕНИИ ПОРОГА ==========
def recalculate_mask(threshold):
    """Пересчет маски и наложения с новым порогом"""
    if st.session_state.ort_session is None or st.session_state.satellite_image is None:
        return
    
    image = st.session_state.satellite_image
    
    # Препроцессинг
    input_tensor = preprocess_image(image)
    ort_inputs = {'input': input_tensor}
    ort_outputs = st.session_state.ort_session.run(['output'], ort_inputs)
    
    # Постпроцессинг с новым порогом
    mask = postprocess_mask(ort_outputs, original_size=image.size, threshold=threshold)
    
    # Создаем наложение
    overlay = create_overlay(image, mask, alpha=0.4)
    
    st.session_state.mask = mask
    st.session_state.overlay_image = overlay
    st.session_state.forest_percent = calculate_forest_percentage(mask)
    st.session_state.current_threshold = threshold

# ========== ИНИЦИАЛИЗАЦИЯ СЕССИИ ==========
if 'analyzed_lat' not in st.session_state:
    st.session_state.analyzed_lat = 55.7558
if 'analyzed_lon' not in st.session_state:
    st.session_state.analyzed_lon = 37.6173
if 'analyzed_zoom' not in st.session_state:
    st.session_state.analyzed_zoom = 14
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False
if 'mask' not in st.session_state:
    st.session_state.mask = None
if 'forest_percent' not in st.session_state:
    st.session_state.forest_percent = 0
if 'satellite_image' not in st.session_state:
    st.session_state.satellite_image = None
if 'overlay_image' not in st.session_state:
    st.session_state.overlay_image = None
if 'ort_session' not in st.session_state:
    st.session_state.ort_session = None
if 'current_threshold' not in st.session_state:
    st.session_state.current_threshold = 0.5
if 'raw_output' not in st.session_state:
    st.session_state.raw_output = None

# ========== ИНТЕРФЕЙС ==========
st.title("🌲 Сегментация леса на спутниковых снимках")
st.markdown("*Перемещайтесь по карте → нажмите кнопку → настройте порог*")

# Две колонки: карта и результат
col_map, col_result = st.columns([1, 1])

with col_map:
    st.subheader("🗺️ Карта")
    
    # Создаем карту
    m = create_map(
        st.session_state.analyzed_lat,
        st.session_state.analyzed_lon,
        st.session_state.analyzed_zoom
    )
    
    # Отображаем карту с фиксированной высотой
    map_data = st_folium(
        m,
        width=None,
        height=550,  # Фиксированная высота карты
        returned_objects=["center", "zoom"],
        key="forest_map"
    )
    
    # Кнопка анализа под картой
    st.markdown("---")
    analyze_btn = st.button(
        "🌲 ЗАФИКСИРОВАТЬ И АНАЛИЗИРОВАТЬ",
        type="primary",
        use_container_width=True
    )
    
    # Информация о позиции
    st.caption(f"📌 Зафиксировано: {st.session_state.analyzed_lat:.4f}, {st.session_state.analyzed_lon:.4f} | 🔍 Zoom: {st.session_state.analyzed_zoom}")

with col_result:
    st.subheader("🛰️ Результат сегментации")
    
    if st.session_state.analyzed and st.session_state.overlay_image:
        # Контейнер для изображения с фиксированной высотой
        img_container = st.container()
        
        with img_container:
            # Отображаем наложение
            st.image(
                st.session_state.overlay_image,
                caption=f"🌲 Лес: {st.session_state.forest_percent:.1f}%",
                use_container_width=True
            )
        
        # Ползунок порога под изображением
        st.markdown("---")
        st.markdown("#### 🎚️ Порог чувствительности")
        st.caption("Выше порог — меньше леса, ниже порог — больше леса")
        
        new_threshold = st.slider(
            "Threshold",
            min_value=0.1,
            max_value=0.9,
            value=st.session_state.current_threshold,
            step=0.05,
            key="threshold_slider",
            label_visibility="collapsed"
        )
        
        # Если порог изменился — пересчитываем маску
        if new_threshold != st.session_state.current_threshold:
            recalculate_mask(new_threshold)
            st.rerun()
        
        # Дополнительная информация
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.metric("🌲 Лес", f"{st.session_state.forest_percent:.1f}%")
        with col_info2:
            st.metric("🎚️ Порог", f"{st.session_state.current_threshold:.2f}")
        with col_info3:
            if st.button("🔄 Сбросить", use_container_width=True):
                st.session_state.analyzed = False
                st.session_state.mask = None
                st.session_state.forest_percent = 0
                st.session_state.satellite_image = None
                st.session_state.overlay_image = None
                st.session_state.raw_output = None
                st.session_state.current_threshold = 0.5
                st.rerun()
        
        # Опционально: показать отдельные слои
        with st.expander("📸 Посмотреть отдельные слои", expanded=False):
            col_sat, col_mask = st.columns(2)
            with col_sat:
                st.image(st.session_state.satellite_image, caption="Спутниковый снимок", use_container_width=True)
            with col_mask:
                st.image(st.session_state.mask, caption="Маска леса", use_container_width=True)
    else:
        # Заглушка с той же высотой что и карта
        placeholder = Image.new('RGB', (512, 512), color=(30, 30, 30))
        st.image(placeholder, use_container_width=True)
        
        st.markdown("---")
        st.caption("Нажмите кнопку под картой для анализа")

# ========== ОБРАБОТКА НАЖАТИЯ КНОПКИ ==========
if analyze_btn:
    # Фиксируем ТЕКУЩИЕ координаты с карты
    if map_data and 'center' in map_data and map_data['center']:
        st.session_state.analyzed_lat = map_data['center']['lat']
        st.session_state.analyzed_lon = map_data['center']['lng']
    
    if map_data and 'zoom' in map_data:
        st.session_state.analyzed_zoom = map_data['zoom']
    
    # Загружаем модель
    with st.spinner("🔄 Загрузка модели..."):
        st.session_state.ort_session = load_model()
    
    if st.session_state.ort_session is None:
        st.error("❌ Не удалось загрузить модель")
    else:
        with st.spinner(f"🛰️ Загрузка спутникового снимка..."):
            image = get_satellite_image(
                st.session_state.analyzed_lat,
                st.session_state.analyzed_lon,
                st.session_state.analyzed_zoom
            )
            
            if image is None:
                st.error("❌ Не удалось загрузить спутниковый снимок")
            else:
                st.session_state.satellite_image = image
                
                with st.spinner("🤖 Сегментация леса..."):
                    start_time = time.time()
                    
                    input_tensor = preprocess_image(image)
                    ort_inputs = {'input': input_tensor}
                    ort_outputs = st.session_state.ort_session.run(['output'], ort_inputs)
                    
                    st.session_state.raw_output = ort_outputs
                    
                    # Используем текущий порог (по умолчанию 0.5)
                    mask = postprocess_mask(
                        ort_outputs, 
                        original_size=image.size, 
                        threshold=st.session_state.current_threshold
                    )
                    
                    inference_time = time.time() - start_time
                
                # Создаем наложение
                overlay = create_overlay(image, mask, alpha=0.4)
                
                st.session_state.mask = mask
                st.session_state.overlay_image = overlay
                st.session_state.forest_percent = calculate_forest_percentage(mask)
                st.session_state.analyzed = True
                
                st.success(f"✅ Анализ завершен за {inference_time:.2f} сек")
                st.rerun()

# ========== ФУТЕР ==========
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; padding: 10px;'>
    <p>🌲 Forest Segmentation App | Модель: U-Net + ResNet34 (IoU: 0.785)</p>
</div>
""", unsafe_allow_html=True)