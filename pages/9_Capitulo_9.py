import streamlit as st
import cv2
import numpy as np
import io

# --- Configuración de la Página ---
st.set_page_config(page_title="Capítulo 9", page_icon="9️⃣", layout="wide")


# --- Clase para el Detector Denso (Implementación del capítulo 9) ---

class DenseDetector():
    """
    Detector de características que genera KeyPoints de forma densa 
    y uniforme sobre una cuadrícula de la imagen.
    """
    def __init__(self, step_size=20, feature_scale=20, img_bound=5):
        # step_size: El diámetro de la KeyPoint (tamaño del círculo dibujado)
        self.initXyStep = step_size
        # feature_scale: Distancia de la cuadrícula (densidad)
        self.initFeatureScale = feature_scale
        # img_bound: Margen interno para evitar los bordes
        self.initImgBound = img_bound
    
    def detect(self, img):
        """Genera KeyPoints en una cuadrícula uniforme."""
        keypoints = []
        rows, cols = img.shape[:2]
        
        # Iterar sobre las filas con el paso definido por feature_scale
        for x in range(self.initImgBound, rows - self.initImgBound, self.initFeatureScale):
            # Iterar sobre las columnas con el paso definido por feature_scale
            for y in range(self.initImgBound, cols - self.initImgBound, self.initFeatureScale):
                # Crear KeyPoint: (coordenada x, coordenada y, tamaño del descriptor)
                keypoints.append(cv2.KeyPoint(float(y), float(x), self.initXyStep))
        return keypoints

# --- Funciones de Detección de Referencia (Dispersa) ---

def sparse_detector_gftt(img):
    """
    Detecta puntos de interés dispersos (esquinas) usando Shi-Tomasi/GFTT.
    Esto simula el comportamiento de detectores 'naturales' como SIFT.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Parámetros GFTT: máximo 500 esquinas, calidad mínima 0.01, distancia euclidiana 10
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=500, qualityLevel=0.01, minDistance=10)
    
    keypoints = []
    if corners is not None:
        for corner in corners:
            x, y = corner.ravel()
            # Usamos un tamaño fijo para la visualización (step_size de 10)
            keypoints.append(cv2.KeyPoint(float(x), float(y), 10))
    return keypoints

# --- Función Principal de la App ---

def run_capitulo9():
    
    # --- Inicialización de Estado ---
    if 'processed_image' not in st.session_state:
        st.session_state.processed_image = False
    
    # --- 1. Títulos y Concepto ---
    st.title("CAPÍTULO 9: Reconocimiento de Objetos")
    st.markdown("###### _Object Recognition_")
    st.markdown("---")

    st.subheader("Visualización de Detectores de Características: Denso vs. Disperso")
    st.info("""
    Para construir un sistema de reconocimiento de objetos (como Bag of Words), necesitamos capturar 
    información de **toda** la imagen, no solo de los puntos más prominentes (esquinas, bordes).
    
    * **Detector Disperso (Sparse):** Solo encuentra los puntos de interés más fuertes. (e.g., SIFT, GFTT).
    * **Detector Denso (Dense):** Muestra puntos de forma uniforme en una cuadrícula, asegurando que todos los descriptores contribuyan al vector de características.
    """)

    # ✅ INICIALIZAR TODAS LAS VARIABLES DE SESSION_STATE
    if 'processed_image' not in st.session_state:
        st.session_state.processed_image = False
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'image_input' not in st.session_state:
        st.session_state.image_input = None
        
    # --- 2. Carga de Imagen y Previsualización ---
    st.header("🖼️ Cargar Imagen de Entrada")
    upload_col, preview_col = st.columns([3, 1])

    with upload_col:
        uploaded_file = st.file_uploader(
            "Selecciona una imagen (PNG, JPG, JPEG)", 
            type=["png", "jpg", "jpeg"], 
            key="uploader"
        )
        if uploaded_file:
            uploaded_file.seek(0)
    
    with preview_col:
        st.markdown("<p style='font-size: 0.8em; margin-bottom: 0px;'>Vista Previa:</p>", unsafe_allow_html=True)
        if uploaded_file is not None:
            st.image(uploaded_file, width=100)
        else:
            st.markdown("<div style='height: 100px; border: 1px dashed #ccc; padding: 5px; text-align: center; line-height: 80px; color: #888;'>Sin imagen</div>", unsafe_allow_html=True)

    st.markdown("---")
    
    # --- 3. Configuración para Detector Denso ---
    st.header("⚙️ Configuración del Detector Denso")
    
    col1, col2 = st.columns(2)
    with col1:
        step_size = st.slider(
            "Tamaño del KeyPoint (Diámetro)", 
            min_value=5, max_value=50, value=20, step=5,
            help="Determina el tamaño del círculo que representa el KeyPoint. Esto influye en la escala del descriptor de característica (e.g., SIFT)."
        )
    with col2:
        feature_scale = st.slider(
            "Escala de la Característica (Densidad / Paso)", 
            min_value=5, max_value=40, value=20, step=5,
            help="Distancia en píxeles entre los KeyPoints muestreados. Un valor más PEQUEÑO significa más puntos (más DENSO)."
        )
        
    img_bound = 5 # Margen fijo para evitar bordes

    # --- 4. Botón de Procesamiento ---
    if st.button("Detectar Características y Comparar", type="primary"):
        if uploaded_file is not None:
            with st.spinner('Detectando puntos de interés...'):
                
                # Cargar y preprocesar imagen
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img_cv2 = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                # Opcional: Redimensionar para un procesamiento más rápido
                max_dim = 600
                height, width = img_cv2.shape[:2]
                if max(height, width) > max_dim:
                    scale = max_dim / max(height, width)
                    new_w, new_h = int(width * scale), int(height * scale)
                    img_cv2 = cv2.resize(img_cv2, (new_w, new_h), interpolation=cv2.INTER_AREA)

                # --- Detección Densa ---
                dense_detector = DenseDetector(step_size=step_size, feature_scale=feature_scale, img_bound=img_bound)
                keypoints_dense = dense_detector.detect(img_cv2)
                
                # Dibujar KeyPoints densos (color: rojo)
                img_dense = np.copy(img_cv2)
                img_dense = cv2.drawKeypoints(
                    img_dense, 
                    keypoints_dense, 
                    None, 
                    color=(0, 0, 255), # BGR: Rojo
                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
                )

                # --- Detección Dispersa (Sparse) ---
                keypoints_sparse = sparse_detector_gftt(img_cv2)
                
                # Dibujar KeyPoints dispersos (color: verde)
                img_sparse = np.copy(img_cv2)
                img_sparse = cv2.drawKeypoints(
                    img_sparse, 
                    keypoints_sparse, 
                    None, 
                    color=(0, 255, 0), # BGR: Verde
                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
                )

                # Almacenar resultados en el estado de sesión
                st.session_state.img_dense = img_dense
                st.session_state.img_sparse = img_sparse
                st.session_state.processed_image = True
                st.success(f'Detección completada. Se encontraron {len(keypoints_dense)} puntos densos y {len(keypoints_sparse)} puntos dispersos.')
        else:
            st.error("Por favor, sube una imagen primero.")
            st.session_state.processed_image = False

    # --- 5. Mostrar Resultados ---
    if st.session_state.processed_image:
        st.markdown("---")
        st.header("Resultados de la Detección de Características")

        col_sparse, col_dense = st.columns(2)
        
        # --- Columna Dispersa ---
        with col_sparse:
            st.caption("Resultado con Detector Disperso (Shi-Tomasi/GFTT - Verde)")
            st.image(cv2.cvtColor(st.session_state.img_sparse, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.markdown("""
            Este método solo encuentra **esquinas** fuertes, dejando grandes áreas de la imagen sin muestrear. 
            Funciona bien para la detección y seguimiento, pero no para crear un **Visual Dictionary** completo.
            """)
        
        # --- Columna Densa ---
        with col_dense:
            st.caption("Resultado con Detector Denso (Custom Grid - Rojo)")
            st.image(cv2.cvtColor(st.session_state.img_dense, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.markdown(f"""
            El detector denso (con paso de **{feature_scale}px**) muestrea la imagen en una cuadrícula uniforme. 
            Esto es crucial para que el modelo Bag of Words extraiga un vector de características (histograma) 
            que **represente toda la imagen**, no solo los puntos de alto contraste.
            """)

run_capitulo9()
