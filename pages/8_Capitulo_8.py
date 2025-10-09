import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# Título de la Página
st.set_page_config(page_title="Capítulo 8", page_icon="8️⃣")


# --- Funciones de Tracking ---
def calculate_region_of_interest(frame, tracking_paths):
    """Calcula la región de interés basada en las rutas de seguimiento"""
    mask = np.zeros_like(frame)
    mask[:] = 255
    for x, y in [np.int32(tp[-1]) for tp in tracking_paths]:
        cv2.circle(mask, (x, y), 6, 0, -1)
    return mask


def add_tracking_paths(frame, tracking_paths, max_corners=500):
    """Agrega nuevos puntos de características para rastrear"""
    mask = calculate_region_of_interest(frame, tracking_paths)
    
    # Extraer buenos puntos de características para rastrear
    feature_points = cv2.goodFeaturesToTrack(
        frame, 
        mask=mask, 
        maxCorners=max_corners,
        qualityLevel=0.3, 
        minDistance=7, 
        blockSize=7
    )
    
    if feature_points is not None:
        for x, y in np.float32(feature_points).reshape(-1, 2):
            tracking_paths.append([(x, y)])


def compute_feature_points(tracking_paths, prev_img, current_img, tracking_params):
    """Calcula el flujo óptico de los puntos de características"""
    feature_points = [tp[-1] for tp in tracking_paths]
    feature_points_0 = np.float32(feature_points).reshape(-1, 1, 2)
    
    # Calcular flujo óptico hacia adelante
    feature_points_1, status_1, err_1 = cv2.calcOpticalFlowPyrLK(
        prev_img, current_img, feature_points_0, None, **tracking_params
    )
    
    # Calcular flujo óptico hacia atrás para verificación
    feature_points_0_rev, status_2, err_2 = cv2.calcOpticalFlowPyrLK(
        current_img, prev_img, feature_points_1, None, **tracking_params
    )
    
    # Calcular la diferencia de los puntos de características
    diff_feature_points = abs(feature_points_0 - feature_points_0_rev).reshape(-1, 2).max(-1)
    
    # Umbral y mantener solo los buenos puntos
    good_points = diff_feature_points < 1
    return feature_points_1.reshape(-1, 2), good_points


# --- Transformador de Video para Tracking ---
class OpticalFlowTracker(VideoTransformerBase):
    """Transformador para seguimiento de flujo óptico"""
    
    def __init__(self):
        self.tracking_paths = []
        self.prev_gray = None
        self.frame_index = 0
        self.num_frames_to_track = 5
        self.num_frames_jump = 2
        self.tracking_params = dict(
            winSize=(11, 11),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        self.max_corners = 500
    
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Convertir a escala de grises
        frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        output_img = img.copy()
        
        if len(self.tracking_paths) > 0:
            prev_img, current_img = self.prev_gray, frame_gray
            
            # Calcular puntos de características usando flujo óptico
            feature_points, good_points = compute_feature_points(
                self.tracking_paths, prev_img, current_img, self.tracking_params
            )
            
            new_tracking_paths = []
            for tp, (x, y), good_points_flag in zip(self.tracking_paths, feature_points, good_points):
                if not good_points_flag:
                    continue
                
                tp.append((x, y))
                
                # Usar estructura de cola: primero en entrar, primero en salir
                if len(tp) > self.num_frames_to_track:
                    del tp[0]
                
                new_tracking_paths.append(tp)
                
                # Dibujar círculos verdes sobre la imagen de salida
                cv2.circle(output_img, (int(x), int(y)), 3, (0, 255, 0), -1)
            
            self.tracking_paths = new_tracking_paths
            
            # Dibujar líneas verdes sobre la imagen de salida
            point_paths = [np.int32(tp) for tp in self.tracking_paths]
            cv2.polylines(output_img, point_paths, False, (0, 150, 0), 2)
        
        # Condición para omitir cada 'n' cuadros
        if not self.frame_index % self.num_frames_jump:
            add_tracking_paths(frame_gray, self.tracking_paths, self.max_corners)
        
        self.frame_index += 1
        self.prev_gray = frame_gray
        
        # Agregar información de seguimiento en la imagen
        text = f"Puntos rastreados: {len(self.tracking_paths)}"
        cv2.putText(output_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 255, 0), 2)
        
        return output_img
    
    def update_params(self, num_frames_to_track, num_frames_jump, max_corners):
        """Actualiza los parámetros de tracking"""
        self.num_frames_to_track = num_frames_to_track
        self.num_frames_jump = num_frames_jump
        self.max_corners = max_corners


def run_capitulo8():
    # ------------------ 1. Títulos ------------------
    st.title("CAPÍTULO 8: Seguimiento de Objetos")
    st.markdown("##### *Object Tracking*")

    st.markdown("---")

    # ------------------ 2. Subtítulo y Concepto ------------------
    st.subheader("Seguimiento Basado en Características | **Feature-based Tracking**")
    st.info("""
    El seguimiento basado en características utiliza el algoritmo de flujo óptico Lucas-Kanade para rastrear 
    puntos de interés a lo largo del tiempo. Identifica características distintivas en la imagen y las sigue 
    fotograma a fotograma, creando trayectorias visuales que muestran el movimiento de los objetos.
    """)


    # ------------------ 4. Configuración de Parámetros ------------------
    st.header("⚙️ Parámetros de Tracking")
    
    col_param1, col_param2, col_param3 = st.columns(3)
    
    with col_param1:
        num_frames_to_track = st.slider(
            "Longitud de Trayectoria",
            min_value=3,
            max_value=20,
            value=5,
            help="Número de fotogramas que permanecen visibles en la trayectoria. Mayor valor = líneas más largas."
        )
    
    with col_param2:
        num_frames_jump = st.slider(
            "Salto de Fotogramas",
            min_value=1,
            max_value=5,
            value=2,
            help="Detecta nuevos puntos cada N fotogramas. Mayor valor = mejor rendimiento pero menos actualización."
        )
    
    with col_param3:
        max_corners = st.slider(
            "Máximo de Puntos",
            min_value=50,
            max_value=1000,
            value=500,
            step=50,
            help="Número máximo de puntos de características a detectar y rastrear."
        )

    st.markdown("---")

    # ------------------ 5. Video en Tiempo Real ------------------
    st.header("🎥 Seguimiento en Tiempo Real")
    
    col_info, col_video = st.columns([1, 2])
    
    with col_info:
        st.subheader("Instrucciones")
        st.markdown("""
        **Cómo usar:**
        1. Haz clic en **START** para iniciar la cámara
        2. Mueve objetos frente a la cámara
        3. Los puntos verdes muestran las características rastreadas
        4. Las líneas verdes muestran las trayectorias de movimiento
        5. Ajusta los parámetros para optimizar el seguimiento
        
        **Consejos:**
        - Objetos con texturas y esquinas se rastrean mejor
        - Movimientos lentos producen mejores resultados
        - Buena iluminación mejora la detección
        """)
        
        st.info("💡 El contador en pantalla muestra cuántos puntos están siendo rastreados activamente.")
    
    with col_video:
        st.subheader("Video con Tracking")
        
        # Configuración WebRTC
        rtc_config = RTCConfiguration({
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        })
        
        # Crear el contexto del transformador
        ctx = webrtc_streamer(
            key="optical-flow-tracking",
            video_transformer_factory=OpticalFlowTracker,
            rtc_configuration=rtc_config,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
        # Actualizar parámetros del transformador
        if ctx.video_transformer:
            ctx.video_transformer.update_params(
                num_frames_to_track, 
                num_frames_jump, 
                max_corners
            )
    
    st.markdown("---")
    
run_capitulo8()