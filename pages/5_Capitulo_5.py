import streamlit as st
import cv2
import numpy as np
import io

# Título de la Página
st.set_page_config(page_title="Capítulo 5", page_icon="5️⃣")


def extract_sift_features(img_input):
    """
    Extrae características SIFT de la imagen.

    Args:
        img_input (np.array): Imagen de entrada en formato BGR (OpenCV).

    Returns:
        tuple: (Imagen con keypoints dibujados, número de keypoints)
    """
    gray_image = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
    
    # Crear detector SIFT
    sift = cv2.SIFT_create()
    keypoints = sift.detect(gray_image, None)
    
    # Dibujar keypoints
    img_output = img_input.copy()
    cv2.drawKeypoints(img_output, keypoints, img_output, 
                     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    return img_output, len(keypoints)


def extract_brief_features(img_input):
    """
    Extrae características usando ORB (alternativa moderna a BRIEF).
    ORB = Oriented FAST and Rotated BRIEF, que es una mejora de BRIEF.

    Args:
        img_input (np.array): Imagen de entrada en formato BGR (OpenCV).

    Returns:
        tuple: (Imagen con keypoints dibujados, número de keypoints, mensaje de error)
    """
    gray_image = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
    
    try:
        # OPCIÓN 1: Intentar usar BRIEF desde xfeatures2d (si está disponible)
        try:
            fast = cv2.FastFeatureDetector_create()
            brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
            keypoints = fast.detect(gray_image, None)
            keypoints, descriptors = brief.compute(gray_image, keypoints)
            method_used = "BRIEF"
        except (AttributeError, cv2.error):
            # OPCIÓN 2: Usar ORB como alternativa (recomendado para OpenCV 4.x)
            orb = cv2.ORB_create(nfeatures=500)  # Limitar a 500 keypoints
            keypoints, descriptors = orb.detectAndCompute(gray_image, None)
            method_used = "ORB (alternativa a BRIEF)"
        
        # Dibujar keypoints
        img_output = img_input.copy()
        cv2.drawKeypoints(img_output, keypoints, img_output, color=(0, 255, 0))
        
        return img_output, len(keypoints), None, method_used
        
    except Exception as e:
        error_msg = f"Error al ejecutar detección de características: {str(e)}"
        return img_input, 0, error_msg, "Ninguno"


def display_feature_tab(tab, img_input, feature_type):
    """Función auxiliar para mostrar el contenido de una pestaña de extracción de características."""
    with tab:
        if st.session_state.processed_image:
            if feature_type == 'sift':
                img_output, num_keypoints = extract_sift_features(img_input)
                display_title = "Salida: SIFT Features"
                error_msg = None
                method_used = "SIFT"
            elif feature_type == 'brief':
                img_output, num_keypoints, error_msg, method_used = extract_brief_features(img_input)
                display_title = f"Salida: {method_used}"
            else:
                img_output = img_input
                num_keypoints = 0
                display_title = "Sin procesamiento"
                error_msg = None
                method_used = "Ninguno"
            
            # Mostrar información del método usado
            if feature_type == 'brief' and method_used == "ORB (alternativa a BRIEF)":
                st.info("ℹ️ **Nota:** Se está usando **ORB** (Oriented FAST and Rotated BRIEF) como alternativa moderna a BRIEF. ORB es más rápido y robusto que BRIEF estándar.")
            
            # Mostrar error si existe
            if error_msg:
                st.error(f"⚠️ {error_msg}")
                st.info("💡 **Alternativa:** Puedes usar SIFT que también está disponible y funciona de manera similar.")
            
            # Usar columnas para mostrar Original vs Procesada
            col_orig, col_out = st.columns(2)
            
            with col_orig:
                st.caption("Entrada Original")
                st.image(cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            with col_out:
                st.caption(display_title)
                st.image(cv2.cvtColor(img_output, cv2.COLOR_BGR2RGB), use_container_width=True)
                if num_keypoints > 0:
                    st.success(f"✅ {num_keypoints} puntos clave detectados con {method_used}")
                elif not error_msg:
                    st.warning("⚠️ No se detectaron puntos clave.")
        else:
            st.warning("Pulsa 'Extraer Características' después de subir una imagen para ver los resultados.")


def run_capitulo5():
    
    # ------------------ 1. Títulos ------------------
    st.title("CAPÍTULO 5: Extrayendo Características de una Imagen")
    st.markdown("##### *Extracting Features from an Image*")

    st.markdown("---")

    # ------------------ 2. Subtítulo y Concepto ------------------
    st.subheader("Scale-invariant feature transform (SIFT) and Binary robust independent elementary features (BRIEF)")
    st.info("""
    **SIFT** y **BRIEF** son algoritmos de detección de características robustas en imágenes, útiles para reconocimiento de objetos, seguimiento y matching de imágenes.
    
    - **SIFT**: Detecta puntos clave invariantes a escala, rotación e iluminación.
    - **BRIEF/ORB**: Descriptores binarios rápidos y eficientes para matching.
    """)
    
    # ------------------ 3. Carga de Imagen y Previsualización ------------------
    st.header("🖼️ Cargar Imagen de Entrada")
    
    upload_col, preview_col = st.columns([3, 1])

    with upload_col:
        uploaded_file = st.file_uploader("Selecciona una imagen (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"], key="uploader_c5")
    
    with preview_col:
        st.markdown("<p style='font-size: 0.8em; margin-bottom: 0px;'>Vista Previa:</p>", unsafe_allow_html=True)
        if uploaded_file is not None:
            st.image(uploaded_file, width=100)
        else:
            st.markdown("<div style='height: 100px; border: 1px dashed #ccc; padding: 5px; text-align: center; line-height: 80px; color: #888;'>Sin imagen</div>", unsafe_allow_html=True)


    # Inicializar el estado de sesión para el control de procesamiento
    if 'processed_image' not in st.session_state:
        st.session_state.processed_image = False

    
    # ------------------ 4. Botón de Procesamiento ------------------
    if st.button("Extraer Características", type="primary"):
        if uploaded_file is not None:
            # IMPORTANTE: Resetear el puntero del archivo al inicio
            uploaded_file.seek(0)
            
            # Lee el archivo subido como bytes
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            
            # Decodifica la imagen en formato OpenCV (BGR)
            img_cv2 = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if img_cv2 is not None:
                # Guarda la imagen procesada en el estado de la sesión
                st.session_state.image_input = img_cv2
                st.session_state.processed_image = True
            else:
                st.error("Error al cargar la imagen. Por favor, intenta con otra imagen.")
                st.session_state.processed_image = False
        else:
            st.error("Por favor, sube una imagen primero.")
            st.session_state.processed_image = False


    # ------------------ 5. Pestañas de Resultados ------------------
    if st.session_state.processed_image:
        img_input = st.session_state.image_input
        
        st.markdown("---")
        st.header("Resultados de la Extracción de Características")

        # Define las pestañas
        tab_sift, tab_brief = st.tabs(["SIFT", "BRIEF/ORB"])

        # Contenido de la Pestaña 1 (SIFT)
        display_feature_tab(tab_sift, img_input, 'sift')

        # Contenido de la Pestaña 2 (BRIEF/ORB)
        display_feature_tab(tab_brief, img_input, 'brief')


run_capitulo5()