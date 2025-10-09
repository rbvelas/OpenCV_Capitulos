import streamlit as st
import cv2
import numpy as np
import io
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# Título de la Página
st.set_page_config(page_title="Capítulo 10", page_icon="🔟", layout="wide")

# --- Clase para Realidad Aumentada ---

class ARTransformer(VideoTransformerBase):
    """Transformador para overlay de pirámide 3D sobre objeto rastreado"""
    def __init__(self):
        self.feature_detector = cv2.ORB_create()
        self.feature_detector.setMaxFeatures(2000)  # Aumentado para mejor rastreo
        
        # Vértices de la pirámide (base cuadrada + vértice superior)
        self.overlay_vertices = np.float32([
            [0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0],  # Base
            [0.5, 0.5, 4]  # Vértice superior
        ])
        
        # Colores para cada cara de la pirámide
        self.color_lines = (0, 0, 0)  # Negro para las aristas
        
        # Estado del rastreo
        self.target_image = None
        self.target_keypoints = None
        self.target_descriptors = None
        self.target_rect = None
        self.tracking_enabled = False
        
        # ROI seleccionada
        self.roi_x = 100
        self.roi_y = 100
        self.roi_w = 200
        self.roi_h = 200
        self.show_roi = True
        
        # Frame actual para captura
        self.current_frame = None
        self.capture_requested = False
        
        # Suavizado de la pirámide para estabilidad
        self.last_quad = None
        self.quad_history = []
        self.max_history = 5  # Promediar últimos 5 frames
        
        # Matcher con mejores parámetros
        flann_params = dict(algorithm=6, table_number=12, key_size=20, multi_probe_level=2)
        self.matcher = cv2.FlannBasedMatcher(flann_params, {})
    
    def set_roi(self, x, y, w, h):
        """Actualiza la ROI desde los controles de Streamlit"""
        self.roi_x = x
        self.roi_y = y
        self.roi_w = w
        self.roi_h = h
    
    def capture_target(self, frame):
        """Captura el objetivo desde el frame actual usando la ROI"""
        x_start = self.roi_x
        y_start = self.roi_y
        x_end = x_start + self.roi_w
        y_end = y_start + self.roi_h
        
        rect = (x_start, y_start, x_end, y_end)
        
        # Detectar características en la imagen
        keypoints, descriptors = self.feature_detector.detectAndCompute(frame, None)
        
        # Filtrar solo keypoints dentro del rectángulo
        filtered_kp = []
        filtered_desc = []
        
        if keypoints and descriptors is not None:
            for kp, desc in zip(keypoints, descriptors):
                x, y = kp.pt
                if x_start <= x <= x_end and y_start <= y <= y_end:
                    filtered_kp.append(kp)
                    filtered_desc.append(desc)
        
        if len(filtered_kp) > 10:
            self.target_image = frame.copy()
            self.target_keypoints = filtered_kp
            self.target_descriptors = np.array(filtered_desc, dtype='uint8')
            self.target_rect = rect
            self.tracking_enabled = True
            self.show_roi = False
            self.matcher.clear()
            self.matcher.add([self.target_descriptors])
            return True, len(filtered_kp)
        return False, 0
    
    def overlay_pyramid(self, img, quad, rect):
        """Dibuja una pirámide 3D sobre el cuadrilátero rastreado"""
        x_start, y_start, x_end, y_end = rect
        
        # Puntos 3D de la región objetivo
        quad_3d = np.float32([
            [x_start, y_start, 0], [x_end, y_start, 0],
            [x_end, y_end, 0], [x_start, y_end, 0]
        ])
        
        h, w = img.shape[:2]
        
        # Matriz de cámara simplificada
        K = np.float64([
            [w, 0, 0.5*(w-1)],
            [0, w, 0.5*(h-1)],
            [0, 0, 1.0]
        ])
        
        dist_coef = np.zeros(4)
        
        try:
            # Resolver PnP para obtener rotación y traslación
            ret, rvec, tvec = cv2.solvePnP(
                objectPoints=quad_3d,
                imagePoints=quad,
                cameraMatrix=K,
                distCoeffs=dist_coef
            )
            
            if not ret:
                return
            
            # Escalar y trasladar vértices de la pirámide
            verts = self.overlay_vertices * [
                (x_end-x_start), (y_end-y_start), -(x_end-x_start)*0.3
            ] + (x_start, y_start, 0)
            
            # Proyectar vértices 3D a 2D
            verts_2d = cv2.projectPoints(
                verts, rvec, tvec, 
                cameraMatrix=K, 
                distCoeffs=dist_coef
            )[0].reshape(-1, 2)
            
            verts_floor = np.int32(verts_2d).reshape(-1, 2)
            
            # Dibujar caras de la pirámide con diferentes colores
            # Base (cyan)
            cv2.drawContours(img, [verts_floor[:4]], -1, (0, 255, 255), -3)
            
            # Cara frontal (verde)
            cv2.drawContours(img, [np.vstack((verts_floor[:2], verts_floor[4:5]))], -1, (0, 255, 0), -3)
            
            # Cara derecha (rojo)
            cv2.drawContours(img, [np.vstack((verts_floor[1:3], verts_floor[4:5]))], -1, (0, 0, 255), -3)
            
            # Cara trasera (azul oscuro)
            cv2.drawContours(img, [np.vstack((verts_floor[2:4], verts_floor[4:5]))], -1, (150, 0, 0), -3)
            
            # Cara izquierda (amarillo)
            cv2.drawContours(img, [np.vstack((verts_floor[3:4], verts_floor[0:1], verts_floor[4:5]))], -1, (0, 255, 255), -3)
            
            # Dibujar aristas en negro
            edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 4), (2, 4), (3, 4)]
            for i, j in edges:
                pt1 = tuple(verts_floor[i])
                pt2 = tuple(verts_floor[j])
                cv2.line(img, pt1, pt2, self.color_lines, 2)
                
        except Exception as e:
            pass
    
    def request_capture(self):
        """Solicita la captura del objetivo en el próximo frame"""
        self.capture_requested = True
    
    def reset_tracking(self):
        """Reinicia el rastreo"""
        self.tracking_enabled = False
        self.show_roi = True
        self.capture_requested = False
        self.target_image = None
        self.target_keypoints = None
        self.target_descriptors = None
        self.last_quad = None
        self.quad_history = []
    
    def smooth_quad(self, quad):
        """Suaviza el cuadrilátero usando promedio temporal"""
        self.quad_history.append(quad)
        
        # Mantener solo los últimos N frames
        if len(self.quad_history) > self.max_history:
            self.quad_history.pop(0)
        
        # Promediar todos los quads en el historial
        if len(self.quad_history) > 0:
            smoothed = np.mean(self.quad_history, axis=0)
            return smoothed.astype(np.float32)
        
        return quad
    
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Guardar frame actual
        self.current_frame = img.copy()
        
        # Si se solicitó captura, procesar
        if self.capture_requested:
            self.capture_requested = False
            success, num_feat = self.capture_target(img)
            if success:
                st.session_state.target_captured = True
                st.session_state.num_features = num_feat
            else:
                st.session_state.capture_error = "No se detectaron suficientes características"
        
        # Mostrar ROI si aún no se ha capturado el objetivo
        if self.show_roi:
            x_start = self.roi_x
            y_start = self.roi_y
            x_end = x_start + self.roi_w
            y_end = y_start + self.roi_h
            
            # Validar límites
            h, w = img.shape[:2]
            x_start = max(0, min(x_start, w-1))
            y_start = max(0, min(y_start, h-1))
            x_end = max(x_start+1, min(x_end, w))
            y_end = max(y_start+1, min(y_end, h))
            
            # Dibujar rectángulo de selección
            cv2.rectangle(img, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
            cv2.putText(img, "Ajusta ROI y presiona 'Capturar Objetivo'", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            return img
        
        # Realizar rastreo si está habilitado
        if not self.tracking_enabled or self.target_descriptors is None:
            return img
        
        try:
            # Detectar características en el frame actual
            keypoints, descriptors = self.feature_detector.detectAndCompute(img, None)
            
            if descriptors is None or len(keypoints) < 10:
                cv2.putText(img, "Rastreando... (sin suficientes puntos)", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                return img
            
            # Matching
            matches = self.matcher.knnMatch(descriptors, k=2)
            
            # Filtrar buenos matches
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
            
            if len(good_matches) < 10:
                cv2.putText(img, "Rastreando... (objeto no visible)", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                return img
            
            # Extraer puntos correspondientes
            src_pts = np.float32([self.target_keypoints[m.trainIdx].pt for m in good_matches])
            dst_pts = np.float32([keypoints[m.queryIdx].pt for m in good_matches])
            
            # Calcular homografía
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if H is None:
                cv2.putText(img, "Rastreando... (calculando pose)", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                return img
            
            # Transformar el rectángulo objetivo
            x_start, y_start, x_end, y_end = self.target_rect
            quad = np.float32([
                [x_start, y_start], [x_end, y_start],
                [x_end, y_end], [x_start, y_end]
            ])
            quad = cv2.perspectiveTransform(quad.reshape(1, -1, 2), H).reshape(-1, 2)
            
            # Dibujar el cuadrilátero rastreado
            cv2.polylines(img, [np.int32(quad)], True, (0, 255, 0), 2)
            
            # Dibujar puntos característicos
            inliers = mask.ravel() == 1
            for pt in dst_pts[inliers]:
                cv2.circle(img, tuple(np.int32(pt)), 2, (0, 255, 0), -1)
            
            # Overlay de la pirámide 3D
            self.overlay_pyramid(img, quad, self.target_rect)
            
            # Información de rastreo
            cv2.putText(img, f"Rastreando: {len(good_matches)} matches", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
        except Exception as e:
            cv2.putText(img, f"Error: {str(e)[:50]}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return img


def run_capitulo10():
    # ------------------ 1. Títulos ------------------
    st.title("CAPÍTULO 10: Realidad Aumentada - Rastreo Estable")
    st.markdown("##### *Augmented Reality*")
    st.markdown("---")
    
    # ------------------ 2. Subtítulo y Concepto ------------------
    st.subheader("Overlay 3D objects on a video | Superponer objetos 3D en un video")
    st.info("La Realidad Aumentada permite superponer objetos virtuales 3D sobre imágenes del mundo real. Se han aplicado filtros de estabilidad para reducir el 'jitter'.")
    
    # Inicializar estados
    if 'target_captured' not in st.session_state:
        st.session_state.target_captured = False
    if 'num_features' not in st.session_state:
        st.session_state.num_features = 0
    if 'capture_error' not in st.session_state:
        st.session_state.capture_error = None
    
    st.markdown("---")
    st.header("🎥 Realidad Aumentada en Tiempo Real")
    
    # --- 1. Controles de ROI (ANCHO COMPLETO - Nueva Posición) ---
    st.markdown("### 🎯 Controles de ROI")
    
    # Sliders de Posición
    col_x, col_y = st.columns(2)
    with col_x:
        roi_x = st.slider("X (horizontal)", 0, 640, 100, 10, key="roi_x")
    with col_y:
        roi_y = st.slider("Y (vertical)", 0, 480, 100, 10, key="roi_y")

    # Sliders de Tamaño
    col_w, col_h = st.columns(2)
    with col_w:
        roi_w = st.slider("Ancho", 50, 400, 200, 10, key="roi_w")
    with col_h:
        roi_h = st.slider("Alto", 50, 400, 200, 10, key="roi_h")

    st.markdown("---") # Separador antes del layout de 2 columnas

    # --- 2. Layout Principal (Instrucciones | Video) ---
    col_instructions, col_video = st.columns([1, 2])
    
    with col_instructions:
        # Instrucciones (Nueva Posición)
        st.markdown("### 📋 Instrucciones y Controles:") 
        st.markdown("""
        1. **Ajusta el rectángulo verde** (ROI) con los sliders superiores.
        2. **Posiciona la ROI** sobre un objeto con **textura alta** y buena iluminación.
        3. **Presiona 'Capturar Objetivo'** para iniciar el rastreo.
        4. **Mueve el objeto** y la pirámide 3D lo seguirá de forma **estable**.
        
        💡 **Mejores objetos:**
        - Libros con texto
        - Cajas con diseños
        - Posters o imágenes
        - Evita superficies lisas
        """)
        
        st.markdown("---")
        
        # Botones de control (Movidos a col_instructions)
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            capture_btn = st.button("📸 Capturar Objetivo", type="primary", use_container_width=True)
        
        with col_btn2:
            reset_btn = st.button("🔄 Reiniciar", use_container_width=True)
        
        if st.session_state.target_captured:
            st.success(f"✅ Objetivo capturado ({st.session_state.num_features} características detectadas)")
        else:
            # Texto requerido por el usuario
            st.info("👆 Ajusta la ROI y captura el objetivo para comenzar")
        
        # Mostrar error si existe
        if st.session_state.capture_error:
            st.error(f"⚠️ {st.session_state.capture_error}")
            st.session_state.capture_error = None
    
    with col_video:
        st.markdown("### 🎬 Video Stream con AR")
        
        # Configuración WebRTC
        rtc_config = RTCConfiguration({
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        })
        
        ctx = webrtc_streamer(
            key="ar-stream-stable",
            video_transformer_factory=ARTransformer,
            rtc_configuration=rtc_config,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
        # Lógica de Interfaz para pasar comandos al Transformador
        if ctx.video_transformer:
            ctx.video_transformer.set_roi(roi_x, roi_y, roi_w, roi_h)
            
            if capture_btn:
                ctx.video_transformer.request_capture()
                st.session_state.capture_error = None
            
            if reset_btn:
                ctx.video_transformer.reset_tracking()
                st.session_state.target_captured = False
                st.session_state.num_features = 0
                st.session_state.capture_error = None
                st.rerun()
        
        # Advertencia de movimiento
        st.warning("Mueva el objetivo de forma lenta para preservar la visualización estable de la pirámide.")
        

run_capitulo10()
