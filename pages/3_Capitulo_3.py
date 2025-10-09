import streamlit as st
import cv2
import numpy as np
import io
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# Título de la Página
st.set_page_config(page_title="Capítulo 3", page_icon="3️⃣")

# --- Funciones de Procesamiento ---
def process_invert_colors(img_cv2, x0, y0, x1, y1):
    """
    Invierte los colores en una Región de Interés (ROI) definida.
    """
    img_output = img_cv2.copy()
    
    # Aseguramos que los puntos sean válidos (ordenados de menor a mayor)
    min_x, max_x = min(x0, x1), max(x0, x1)
    min_y, max_y = min(y0, y1), max(y0, y1)
    
    # Aseguramos que los límites no excedan las dimensiones de la imagen
    rows, cols = img_cv2.shape[:2]
    min_x = max(0, min_x)
    max_x = min(cols, max_x)
    min_y = max(0, min_y)
    max_y = min(rows, max_y)
    
    # Invertir la región seleccionada (255 - valor_actual)
    if max_x > min_x and max_y > min_y:
        img_output[min_y:max_y, min_x:max_x] = 255 - img_output[min_y:max_y, min_x:max_x]
    
    return img_output

def cartoonize_image(img, ksize=5, sketch_mode=False):
    """
    Aplica el efecto de caricatura o sketch a la imagen.
    """
    num_repetitions, sigma_color, sigma_space, ds_factor = 10, 5, 7, 4
    
    # Convert image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply median filter to the grayscale image
    img_gray = cv2.medianBlur(img_gray, 7)
    
    # Detect edges in the image and threshold it
    edges = cv2.Laplacian(img_gray, cv2.CV_8U, ksize=ksize)
    ret, mask = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY_INV)
    
    # 'mask' is the sketch of the image
    if sketch_mode:
        # Devuelve el sketch (escala de grises)
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # --- Aplicar filtro Bilateral y Combinación ---
    # Resize the image to a smaller size for faster computation
    img_small = cv2.resize(img, None, fx=1.0/ds_factor, fy=1.0/ds_factor, 
                          interpolation=cv2.INTER_AREA)
    
    # Apply bilateral filter the image multiple times
    for i in range(num_repetitions):
        img_small = cv2.bilateralFilter(img_small, ksize, sigma_color, sigma_space)
    
    img_output = cv2.resize(img_small, None, fx=ds_factor, fy=ds_factor, 
                           interpolation=cv2.INTER_LINEAR)
    
    # Add the thick boundary lines to the image using 'AND' operator
    # Redimensionar la máscara al tamaño de img_output para que coincidan
    rows_out, cols_out = img_output.shape[:2]
    mask_resized = cv2.resize(mask, (cols_out, rows_out), interpolation=cv2.INTER_NEAREST)
    # La máscara debe ser de un solo canal (escala de grises)
    dst = cv2.bitwise_and(img_output, img_output, mask=mask_resized)
    
    return dst

# --- Transformadores de Video ---
class InvertROITransformer(VideoTransformerBase):
    """Transformador que invierte colores en una ROI"""
    def __init__(self):
        self.x0 = 0
        self.y0 = 0
        self.x1 = 640
        self.y1 = 480
    
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Aplicar inversión de colores
        result = process_invert_colors(img, self.x0, self.y0, self.x1, self.y1)
        
        # Dibujar rectángulo para mostrar la ROI
        cv2.rectangle(result, (self.x0, self.y0), (self.x1, self.y1), 
                     (0, 255, 0), 2)
        
        return result

class CartoonTransformer(VideoTransformerBase):
    """Transformador que aplica efecto de caricatura"""
    def __init__(self):
        self.sketch_mode = False
        self.ksize = 5
    
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Aplicar efecto de caricatura
        result = cartoonize_image(img, ksize=self.ksize, sketch_mode=self.sketch_mode)
        
        return result

def get_image_download_link(img_cv2, filename, text):
    """Convierte una imagen OpenCV a bytes para la descarga."""
    is_success, buffer = cv2.imencode(".png", img_cv2)
    if is_success:
        bio = io.BytesIO(buffer.tobytes())
        return st.download_button(
            label=text,
            data=bio.getvalue(),
            file_name=filename,
            mime="image/png"
        )
    return None

# --- Función Principal ---
def run_capitulo3():
    # ------------------ 1. Títulos ------------------ 
    st.title("CAPÍTULO 3: Caricaturización y Procesamiento de Video")
    st.markdown("##### *Cartoonizing an Image*")
    st.markdown("---")
    
    # ------------------ 2. Subtítulos y Concepto ------------------ 
    st.subheader("Interacción con Video en Vivo y Efectos de Imagen")
    st.info("Este capítulo explora cómo capturar datos de video en tiempo real y aplicar transformaciones complejas como la caricaturización.")
    
    # ------------------ 3. Modo de Entrada ------------------
    input_mode = st.radio(
        "Selecciona la fuente de imagen:",
        ["Cámara Web (Stream en Vivo)", "Subir Imagen"],
        horizontal=True,
        key="input_mode"
    )
    
    st.markdown("---")
    
    # ==================== MODO: Cámara Web (LIVE) ====================
    if input_mode == "Cámara Web (Stream en Vivo)":
        st.markdown("#### 🎥 Video en Vivo y Procesamiento en Tiempo Real")
        
        # ------------------ Tabs para diferentes efectos ------------------
        tab1, tab2 = st.tabs(["🎨 Invertir Colores en ROI", "🖼️ Efecto Caricatura"])
        
        # --- TAB 1: Invertir Colores en Región (ROI) ---
        with tab1:
            st.header("Invertir Colores en Región (ROI)")
            st.markdown("Ajusta la región rectangular donde se invertirán los colores en tiempo real.")
            
            col_controls, col_video = st.columns([1, 2])
            
            with col_controls:
                st.subheader("Controles de ROI")
                
                # Controles para definir la ROI
                st.markdown("**Esquina Superior Izquierda:**")
                x0 = st.slider("X inicial (x0)", 0, 640, 50, key="roi_x0")
                y0 = st.slider("Y inicial (y0)", 0, 480, 50, key="roi_y0")
                
                st.markdown("**Esquina Inferior Derecha:**")
                x1 = st.slider("X final (x1)", 0, 640, 590, key="roi_x1")
                y1 = st.slider("Y final (y1)", 0, 480, 430, key="roi_y1")
                
                st.info("💡 El rectángulo verde muestra la región donde se invertirán los colores.")
            
            with col_video:
                st.subheader("Video en Vivo")
                
                # Configuración WebRTC
                rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
                
                # Crear el contexto del transformador
                ctx = webrtc_streamer(
                    key="invert-roi",
                    video_transformer_factory=InvertROITransformer,
                    rtc_configuration=rtc_config,
                    media_stream_constraints={"video": True, "audio": False},
                )
                
                # Actualizar parámetros del transformador en tiempo real
                if ctx.video_transformer:
                    ctx.video_transformer.x0 = x0
                    ctx.video_transformer.y0 = y0
                    ctx.video_transformer.x1 = x1
                    ctx.video_transformer.y1 = y1
            
            st.markdown("---")
            
        # --- TAB 2: Efecto Caricatura ---
        with tab2:
            st.header("Efecto Caricatura")
            st.markdown("Transforma el video en tiempo real con efectos de caricatura o sketch.")
            
            col_controls2, col_video2 = st.columns([1, 2])
            
            with col_controls2:
                st.subheader("Controles de Efecto")
                
                # Selector de modo
                mode = st.radio(
                    "Modo de Caricaturización",
                    ["Caricatura a Color", "Sketch (Sin Color)"],
                    key="cartoon_mode"
                )
                
                sketch_mode = (mode == "Sketch (Sin Color)")
                
                # Control de intensidad de bordes
                ksize = st.slider(
                    "Tamaño de Kernel (Intensidad de Bordes)",
                    1, 11, 5, 2,
                    key="cartoon_ksize",
                    help="Valores más altos = bordes más gruesos"
                )
                
                st.info("💡 Los cambios se aplican en tiempo real sobre el video.")
            
            with col_video2:
                st.subheader("Video en Vivo")
                
                # Configuración WebRTC
                rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
                
                # Crear el contexto del transformador
                ctx2 = webrtc_streamer(
                    key="cartoon-effect",
                    video_transformer_factory=CartoonTransformer,
                    rtc_configuration=rtc_config,
                    media_stream_constraints={"video": True, "audio": False},
                )
                
                # Actualizar parámetros del transformador en tiempo real
                if ctx2.video_transformer:
                    ctx2.video_transformer.sketch_mode = sketch_mode
                    ctx2.video_transformer.ksize = ksize
            
            st.markdown("---")
            
    # ==================== MODO: Subir Imagen (estática) ====================
    elif input_mode == "Subir Imagen":
        st.markdown("#### 🖼️ Procesamiento de Imagen Estática")
        
        # Selector de fuente de imagen
        image_source = st.radio(
            "Selecciona la fuente:",
            ["Subir Archivo", "Capturar con Cámara"],
            horizontal=True,
            key="image_source"
        )
        
        img_input_cv2 = None
        
        # --- Opción 1: Subir Archivo ---
        if image_source == "Subir Archivo":
            upload_col, preview_col = st.columns([3, 1])
            
            with upload_col:
                uploaded_file = st.file_uploader(
                    "Selecciona una imagen (PNG, JPG, JPEG)", 
                    type=["png", "jpg", "jpeg"], 
                    key="uploader_c3"
                )
            
            if uploaded_file is not None:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img_input_cv2 = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                with preview_col:
                    st.markdown("<p style='font-size: 0.8em; margin-bottom: 0px;'>Vista Previa:</p>", 
                               unsafe_allow_html=True)
                    st.image(uploaded_file, width=100)
        
        # --- Opción 2: Captura con Cámara ---
        elif image_source == "Capturar con Cámara":
            camera_file = st.camera_input("Toma una foto:")
            
            if camera_file is not None:
                file_bytes = np.asarray(bytearray(camera_file.read()), dtype=np.uint8)
                img_input_cv2 = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img_input_cv2 is None:
            st.info("Sube una imagen o captura una foto para aplicar los efectos.")
            return
        
        st.markdown("---")
        st.header("Aplicar Procesamientos (Imagen Estática)")
        
        col_orig, col_invert, col_cartoon = st.columns(3)
        
        # --- Columna 1: Imagen Original ---
        with col_orig:
            st.markdown("#### 1. Imagen Original")
            st.image(img_input_cv2, channels="BGR", caption="Entrada", use_container_width=True)
            get_image_download_link(img_input_cv2, "original_image.png", "⬇️ Descargar Original")
        
        # --- Columna 2: Invertir Colores en ROI ---
        with col_invert:
            st.markdown("#### 2. Invertir Colores en ROI")
            rows, cols = img_input_cv2.shape[:2]
            
            c1, c2 = st.columns(2)
            with c1:
                x0 = st.number_input("Coordenada X inicial (x0)", 0, cols, value=0, key="static_x0")
            with c2:
                y0 = st.number_input("Coordenada Y inicial (y0)", 0, rows, value=0, key="static_y0")
            
            c3, c4 = st.columns(2)
            with c3:
                x1 = st.number_input("Coordenada X final (x1)", 0, cols, value=cols, key="static_x1")
            with c4:
                y1 = st.number_input("Coordenada Y final (y1)", 0, rows, value=rows, key="static_y1")
            
            # Procesar imagen
            img_inverted = process_invert_colors(img_input_cv2, x0, y0, x1, y1)
            st.image(img_inverted, channels="BGR", caption="Resultado: Inversión de Colores en ROI", 
                    use_container_width=True)
            get_image_download_link(img_inverted, "inverted_roi_image_static.png", "⬇️ Descargar ROI")
        
        # --- Columna 3: Efecto Caricatura ---
        with col_cartoon:
            st.markdown("#### 3. Efecto Caricatura")
            
            static_mode = st.radio(
                "Modo de Caricaturización", 
                ["Caricatura a Color", "Sketch (Sin Color)"], 
                horizontal=True, 
                key="static_mode"
            )
            
            sketch_mode = (static_mode == "Sketch (Sin Color)")
            
            # Procesar imagen
            img_cartoon = cartoonize_image(img_input_cv2, sketch_mode=sketch_mode)
            st.image(img_cartoon, channels="BGR", caption=f"Resultado: {static_mode}", 
                    use_container_width=True)
            get_image_download_link(img_cartoon, "cartoonized_image_static.png", "⬇️ Descargar Caricatura")

run_capitulo3()