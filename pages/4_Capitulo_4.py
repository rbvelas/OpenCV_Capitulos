import streamlit as st
import cv2
import numpy as np
import io
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import os

# Título de la Página
st.set_page_config(page_title="Capítulo 4", page_icon="4️⃣")

# --- Funciones de Detección ---
def detect_faces(frame, face_cascade, apply_filter=False, face_mask=None):
    """Detecta rostros y opcionalmente aplica una máscara"""
    scaling_factor = 0.5
    frame_resized = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, 
                               interpolation=cv2.INTER_AREA)
    
    face_rects = face_cascade.detectMultiScale(frame_resized, scaleFactor=1.3, minNeighbors=3)
    
    if not apply_filter or face_mask is None:
        # Solo dibujar rectángulos
        for (x, y, w, h) in face_rects:
            cv2.rectangle(frame_resized, (x, y), (x+w, y+h), (0, 255, 0), 3)
    else:
        # Aplicar máscara de Hannibal
        h_mask, w_mask = face_mask.shape[:2]
        for (x, y, w, h) in face_rects:
            if h <= 0 or w <= 0:
                continue
            
            # Ajustar dimensiones
            h_adj, w_adj = int(1.4*h), int(1.0*w)
            y_adj = y - int(0.1*h_adj)
            x_adj = int(x)
            
            # Validar límites
            if y_adj < 0 or x_adj < 0:
                continue
            if y_adj + h_adj > frame_resized.shape[0] or x_adj + w_adj > frame_resized.shape[1]:
                continue
            
            # Extraer ROI
            frame_roi = frame_resized[y_adj:y_adj+h_adj, x_adj:x_adj+w_adj]
            face_mask_small = cv2.resize(face_mask, (w_adj, h_adj), interpolation=cv2.INTER_AREA)
            
            # Crear máscaras
            gray_mask = cv2.cvtColor(face_mask_small, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(gray_mask, 180, 255, cv2.THRESH_BINARY_INV)
            mask_inv = cv2.bitwise_not(mask)
            
            try:
                masked_face = cv2.bitwise_and(face_mask_small, face_mask_small, mask=mask)
                masked_frame = cv2.bitwise_and(frame_roi, frame_roi, mask=mask_inv)
                frame_resized[y_adj:y_adj+h_adj, x_adj:x_adj+w_adj] = cv2.add(masked_face, masked_frame)
            except cv2.error as e:
                print(f'Error aplicando máscara: {e}')
                continue
    
    return frame_resized

def detect_eyes(frame, face_cascade, eye_cascade, apply_filter=False, sunglasses_img=None):
    """Detecta ojos y opcionalmente aplica lentes de sol"""
    ds_factor = 0.5
    frame_resized = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor, 
                               interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=1)
    
    if not apply_filter or sunglasses_img is None:
        # Solo dibujar círculos en los ojos
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame_resized[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            
            for (x_eye, y_eye, w_eye, h_eye) in eyes:
                center = (int(x_eye + 0.5*w_eye), int(y_eye + 0.5*h_eye))
                radius = int(0.3 * (w_eye + h_eye))
                cv2.circle(roi_color, center, radius, (0, 255, 0), 3)
    else:
        # Aplicar lentes de sol
        centers = []
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            
            for (x_eye, y_eye, w_eye, h_eye) in eyes:
                centers.append((x + int(x_eye + 0.5*w_eye), y + int(y_eye + 0.5*h_eye)))
        
        if len(centers) >= 2:
            h_sg, w_sg = sunglasses_img.shape[:2]
            eye_distance = abs(centers[1][0] - centers[0][0])
            sunglasses_width = 2.12 * eye_distance
            scaling_factor = sunglasses_width / w_sg
            
            overlay_sunglasses = cv2.resize(sunglasses_img, None, fx=scaling_factor, 
                                           fy=scaling_factor, interpolation=cv2.INTER_AREA)
            
            x_sg = min(centers[0][0], centers[1][0])
            y_sg = centers[0][1]
            
            x_sg -= int(0.26*overlay_sunglasses.shape[1])
            y_sg -= int(0.26*overlay_sunglasses.shape[0])
            
            h_sg, w_sg = overlay_sunglasses.shape[:2]
            
            # Validar límites
            if y_sg < 0 or x_sg < 0:
                return frame_resized
            if y_sg + h_sg > frame_resized.shape[0] or x_sg + w_sg > frame_resized.shape[1]:
                return frame_resized
            
            frame_roi = frame_resized[y_sg:y_sg+h_sg, x_sg:x_sg+w_sg]
            gray_overlay = cv2.cvtColor(overlay_sunglasses, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(gray_overlay, 180, 255, cv2.THRESH_BINARY_INV)
            mask_inv = cv2.bitwise_not(mask)
            
            try:
                masked_sunglasses = cv2.bitwise_and(overlay_sunglasses, overlay_sunglasses, mask=mask)
                masked_frame = cv2.bitwise_and(frame_roi, frame_roi, mask=mask_inv)
                frame_resized[y_sg:y_sg+h_sg, x_sg:x_sg+w_sg] = cv2.add(masked_sunglasses, masked_frame)
            except cv2.error as e:
                print(f'Error aplicando lentes: {e}')
    
    return frame_resized

# --- Transformadores de Video ---
class FaceDetectionTransformer(VideoTransformerBase):
    """Transformador para detección de rostros"""
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml'
        )
        self.apply_filter = False
        self.face_mask = None
    
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        result = detect_faces(img, self.face_cascade, self.apply_filter, self.face_mask)
        return result

class EyeDetectionTransformer(VideoTransformerBase):
    """Transformador para detección de ojos"""
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        self.apply_filter = False
        self.sunglasses_img = None
    
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        result = detect_eyes(img, self.face_cascade, self.eye_cascade, 
                           self.apply_filter, self.sunglasses_img)
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
def run_capitulo4():
    # ------------------ 1. Títulos ------------------ 
    st.title("CAPÍTULO 4: Detecting and Tracking Different Body Parts")
    st.markdown("##### *Detecting and tracking faces and Detecting eyes*")
    st.markdown("---")
    
    # ------------------ 2. Subtítulos y Concepto ------------------ 
    st.subheader("Detección de Rostros y Ojos con Haar Cascades")
    st.info("Este capítulo utiliza clasificadores Haar Cascade para detectar rostros y ojos en tiempo real, con opción de aplicar filtros divertidos.")
    
    # ------------------ 3. Modo de Entrada ------------------
    input_mode = st.radio(
        "Selecciona la fuente de imagen:",
        ["Cámara Web (Stream en Vivo)", "Subir Imagen"],
        horizontal=True,
        key="input_mode_c4"
    )
    
    st.markdown("---")
    
    # ==================== MODO: Cámara Web (LIVE) ====================
    if input_mode == "Cámara Web (Stream en Vivo)":
        st.markdown("#### 🎥 Video en Vivo y Detección en Tiempo Real")
        
        # ------------------ Tabs para diferentes detectores ------------------
        tab1, tab2 = st.tabs(["👤 Detección de Rostros", "👁️ Detección de Ojos"])
        
        # --- TAB 1: Detección de Rostros ---
        with tab1:
            st.header("Detección de Rostros")
            st.markdown("Detecta rostros en tiempo real usando Haar Cascade Classifier.")
            
            col_controls, col_video = st.columns([1, 2])
            
            with col_controls:
                st.subheader("Controles")
                
                # Checkbox para activar filtro
                apply_face_filter = st.checkbox(
                    "Aplicar Máscara de Hannibal",
                    value=False,
                    key="face_filter"
                )
                
                # Subir imagen de máscara personalizada
                mask_file = st.file_uploader(
                    "O sube tu propia máscara (PNG con fondo negro):",
                    type=["png"],
                    key="mask_uploader"
                )
                
                if apply_face_filter:
                    st.info("💡 La máscara se ajustará automáticamente al rostro detectado.")
                else:
                    st.info("💡 Se dibujarán rectángulos verdes alrededor de los rostros.")
            
            with col_video:
                st.subheader("Video en Vivo")
                
                # Configuración WebRTC
                rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
                
                # Crear el contexto del transformador
                ctx_face = webrtc_streamer(
                    key="face-detection",
                    video_transformer_factory=FaceDetectionTransformer,
                    rtc_configuration=rtc_config,
                    media_stream_constraints={"video": True, "audio": False},
                )
                
                # Actualizar parámetros del transformador
                if ctx_face.video_transformer:
                    ctx_face.video_transformer.apply_filter = apply_face_filter
                    
                    # Cargar máscara personalizada o usar mask_hannibal.png por defecto
                    if mask_file is not None:
                        file_bytes = np.asarray(bytearray(mask_file.read()), dtype=np.uint8)
                        ctx_face.video_transformer.face_mask = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    elif apply_face_filter:
                        # Intentar cargar mask_hannibal.png
                        if os.path.exists('./images/mask_hannibal.png'):
                            ctx_face.video_transformer.face_mask = cv2.imread('./images/mask_hannibal.png')
                        elif os.path.exists('images/mask_hannibal.png'):
                            ctx_face.video_transformer.face_mask = cv2.imread('images/mask_hannibal.png')
                        elif os.path.exists('mask_hannibal.png'):
                            ctx_face.video_transformer.face_mask = cv2.imread('mask_hannibal.png')
                        else:
                            st.warning("⚠️ No se encontró 'mask_hannibal.png'. Coloca la imagen en la carpeta 'images/'")
                            # Crear una máscara de ejemplo como fallback
                            example_mask = np.zeros((200, 200, 3), dtype=np.uint8)
                            cv2.rectangle(example_mask, (20, 20), (180, 180), (255, 255, 255), -1)
                            cv2.rectangle(example_mask, (40, 40), (160, 160), (0, 0, 0), -1)
                            ctx_face.video_transformer.face_mask = example_mask
            
            st.markdown("---")
        
        # --- TAB 2: Detección de Ojos ---
        with tab2:
            st.header("Detección de Ojos")
            st.markdown("Detecta ojos en tiempo real y opcionalmente aplica lentes de sol.")
            
            col_controls2, col_video2 = st.columns([1, 2])
            
            with col_controls2:
                st.subheader("Controles")
                
                # Checkbox para activar filtro
                apply_eye_filter = st.checkbox(
                    "Aplicar Lentes de Sol",
                    value=False,
                    key="eye_filter"
                )
                
                # Subir imagen de lentes personalizada
                sunglasses_file = st.file_uploader(
                    "O sube tus propios lentes (PNG con fondo negro):",
                    type=["png"],
                    key="sunglasses_uploader"
                )
                
                if apply_eye_filter:
                    st.info("💡 Los lentes se ajustarán automáticamente a la distancia entre los ojos.")
                else:
                    st.info("💡 Se dibujarán círculos verdes alrededor de los ojos.")
            
            with col_video2:
                st.subheader("Video en Vivo")
                
                # Configuración WebRTC
                rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
                
                # Crear el contexto del transformador
                ctx_eye = webrtc_streamer(
                    key="eye-detection",
                    video_transformer_factory=EyeDetectionTransformer,
                    rtc_configuration=rtc_config,
                    media_stream_constraints={"video": True, "audio": False},
                )
                
                # Actualizar parámetros del transformador
                if ctx_eye.video_transformer:
                    ctx_eye.video_transformer.apply_filter = apply_eye_filter
                    
                    # Cargar lentes personalizados o usar sunglasses.png por defecto
                    if sunglasses_file is not None:
                        file_bytes = np.asarray(bytearray(sunglasses_file.read()), dtype=np.uint8)
                        ctx_eye.video_transformer.sunglasses_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    elif apply_eye_filter:
                        # Intentar cargar sunglasses.png
                        if os.path.exists('./images/sunglasses.png'):
                            ctx_eye.video_transformer.sunglasses_img = cv2.imread('./images/sunglasses.png')
                        elif os.path.exists('images/sunglasses.png'):
                            ctx_eye.video_transformer.sunglasses_img = cv2.imread('images/sunglasses.png')
                        elif os.path.exists('sunglasses.png'):
                            ctx_eye.video_transformer.sunglasses_img = cv2.imread('sunglasses.png')
                        else:
                            st.warning("⚠️ No se encontró 'sunglasses.png'. Coloca la imagen en la carpeta 'images/'")
                            # Crear lentes de ejemplo como fallback
                            example_sunglasses = np.zeros((60, 200, 3), dtype=np.uint8)
                            cv2.rectangle(example_sunglasses, (10, 15), (80, 45), (50, 50, 50), -1)
                            cv2.rectangle(example_sunglasses, (120, 15), (190, 45), (50, 50, 50), -1)
                            ctx_eye.video_transformer.sunglasses_img = example_sunglasses
            
            st.markdown("---")
    
    # ==================== MODO: Subir Imagen (estática) ====================
    elif input_mode == "Subir Imagen":
        st.markdown("#### 🖼️ Procesamiento de Imagen Estática")
        
        # Selector de fuente de imagen
        image_source = st.radio(
            "Selecciona la fuente:",
            ["Subir Archivo", "Capturar con Cámara"],
            horizontal=True,
            key="image_source_c4"
        )
        
        img_input_cv2 = None
        
        # --- Opción 1: Subir Archivo ---
        if image_source == "Subir Archivo":
            upload_col, preview_col = st.columns([3, 1])
            
            with upload_col:
                uploaded_file = st.file_uploader(
                    "Selecciona una imagen (PNG, JPG, JPEG)", 
                    type=["png", "jpg", "jpeg"], 
                    key="uploader_c4"
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
            st.info("Sube una imagen o captura una foto para aplicar la detección.")
            return
        
        st.markdown("---")
        st.header("Detección en Imagen Estática")
        
        # Cargar cascades
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        col_faces, col_eyes = st.columns(2)
        
        # --- Columna 1: Detección de Rostros ---
        with col_faces:
            st.markdown("#### 👤 Detección de Rostros")
            
            apply_face_mask = st.checkbox("Aplicar Máscara", value=False, key="static_face_mask")
            
            face_mask_static = None
            if apply_face_mask:
                mask_file_static = st.file_uploader(
                    "Sube una máscara personalizada (PNG):",
                    type=["png"],
                    key="static_mask_file"
                )
                if mask_file_static:
                    mask_bytes = np.asarray(bytearray(mask_file_static.read()), dtype=np.uint8)
                    face_mask_static = cv2.imdecode(mask_bytes, cv2.IMREAD_COLOR)
                else:
                    # Cargar mask_hannibal.png por defecto
                    if os.path.exists('./images/mask_hannibal.png'):
                        face_mask_static = cv2.imread('./images/mask_hannibal.png')
                    elif os.path.exists('images/mask_hannibal.png'):
                        face_mask_static = cv2.imread('images/mask_hannibal.png')
                    elif os.path.exists('mask_hannibal.png'):
                        face_mask_static = cv2.imread('mask_hannibal.png')
                    else:
                        st.warning("⚠️ No se encontró 'mask_hannibal.png'")
            
            img_faces = detect_faces(img_input_cv2, face_cascade, apply_face_mask, face_mask_static)
            st.image(img_faces, channels="BGR", caption="Rostros Detectados", use_container_width=True)
            get_image_download_link(img_faces, "faces_detected.png", "⬇️ Descargar Rostros")
        
        # --- Columna 2: Detección de Ojos ---
        with col_eyes:
            st.markdown("#### 👁️ Detección de Ojos")
            
            apply_sunglasses_static = st.checkbox("Aplicar Lentes", value=False, key="static_sunglasses")
            
            sunglasses_static = None
            if apply_sunglasses_static:
                sunglasses_file_static = st.file_uploader(
                    "Sube unos lentes personalizados (PNG):",
                    type=["png"],
                    key="static_sunglasses_file"
                )
                if sunglasses_file_static:
                    sg_bytes = np.asarray(bytearray(sunglasses_file_static.read()), dtype=np.uint8)
                    sunglasses_static = cv2.imdecode(sg_bytes, cv2.IMREAD_COLOR)
                else:
                    # Cargar sunglasses.png por defecto
                    if os.path.exists('./images/sunglasses.png'):
                        sunglasses_static = cv2.imread('./images/sunglasses.png')
                    elif os.path.exists('images/sunglasses.png'):
                        sunglasses_static = cv2.imread('images/sunglasses.png')
                    elif os.path.exists('sunglasses.png'):
                        sunglasses_static = cv2.imread('sunglasses.png')
                    else:
                        st.warning("⚠️ No se encontró 'sunglasses.png'")
            
            img_eyes = detect_eyes(img_input_cv2, face_cascade, eye_cascade, 
                                  apply_sunglasses_static, sunglasses_static)
            st.image(img_eyes, channels="BGR", caption="Ojos Detectados", use_container_width=True)
            get_image_download_link(img_eyes, "eyes_detected.png", "⬇️ Descargar Ojos")

run_capitulo4()