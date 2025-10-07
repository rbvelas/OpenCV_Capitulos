import streamlit as st
import cv2
import numpy as np
import io

# Título de la Página
st.set_page_config(page_title="Capítulo 2", page_icon="2️⃣")


def process_filter(img_input_gray, filter_type):
    """
    Aplica filtros y detectores de borde a la imagen de entrada en escala de grises.

    Args:
        img_input_gray (np.array): Imagen de entrada en formato escala de grises (OpenCV).
        filter_type (str): Tipo de filtro a aplicar ('laplacian', 'canny').

    Returns:
        np.array: Imagen transformada.
        str: Título de la salida.
    """
    
    if filter_type == 'laplacian':
        # Detector de bordes Laplacian. Usamos cv2.CV_64F
        laplacian = cv2.Laplacian(img_input_gray, cv2.CV_64F)
        
        # Normalizar para visualización
        laplacian_norm = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        return laplacian_norm, "Salida: Filtro Laplaciano (Normalizado)"
        
    elif filter_type == 'canny':
        # Detector de bordes Canny (uint8). Requiere dos umbrales (thresholds)
        # Los umbrales (50 y 240) se pueden ajustar para obtener diferentes resultados
        canny = cv2.Canny(img_input_gray, 50, 240)
        
        # Canny devuelve una imagen binaria (blanco y negro), no requiere normalización
        return canny, "Salida: Detección de Bordes Canny (Umbrales: 50, 240)"
        
    return None, "Error de procesamiento"


def display_laplacian_canny_tab(tab, img_input_gray):
    """Función para mostrar los resultados combinados de Laplacian y Canny."""
    with tab:
        if st.session_state.get('processed_image_c2'):
            # 1. Obtener Laplacian y Canny
            laplacian_out, _ = process_filter(img_input_gray, 'laplacian')
            canny_out, _ = process_filter(img_input_gray, 'canny')
            
            st.subheader("Comparación de Laplaciano y Canny")

            # Usar tres columnas: Original, Laplaciano, Canny
            col_orig, col_laplacian, col_canny = st.columns(3)
            
            with col_orig:
                st.caption("Entrada Original (Escala de Grises)")
                st.image(img_input_gray, use_container_width=True, channels="GRAY")
            
            with col_laplacian:
                st.caption("Filtro Laplaciano")
                st.image(laplacian_out, use_container_width=True, channels="GRAY")
            
            with col_canny:
                st.caption("Detector de Bordes Canny")
                st.image(canny_out, use_container_width=True, channels="GRAY")
        else:
            st.warning("Pulsa 'Aplicar Filtros' después de subir una imagen para ver los resultados.")


def display_single_output_tab(tab, img_input_gray, filter_type, title):
    """Función para mostrar un único resultado de filtro (Original vs Salida)."""
    with tab:
        if st.session_state.get('processed_image_c2'):
            img_output, display_title = process_filter(img_input_gray, filter_type)
            
            st.subheader(title)

            # Usar dos columnas para mostrar Original vs Transformada
            col_orig, col_out = st.columns(2)
            
            with col_orig:
                st.caption("Entrada Original (Escala de Grises)")
                # La imagen de entrada debe mostrarse en escala de grises ya que el filtro la usa así
                st.image(img_input_gray, use_container_width=True, channels="GRAY")
            with col_out:
                st.caption(display_title)
                st.image(img_output, use_container_width=True, channels="GRAY")
        else:
            st.warning("Pulsa 'Aplicar Filtros' después de subir una imagen para ver los resultados.")


def run_capitulo2():
    
    # ------------------ 1. Títulos ------------------
    st.title("CAPÍTULO 2: Detección de Bordes y Aplicación de Filtros de Imagen")
    st.markdown("##### *Detecting Edges and Applying Image Filters*")

    st.markdown("---")

    # ------------------ 2. Subtítulo y Concepto ------------------
    st.subheader("Detección de Bordes | **Edge Detection**")
    st.info("La detección de bordes permite identificar las discontinuidades de intensidad en una imagen, esenciales para el reconocimiento de objetos y el análisis de visión por computadora.")
    
    # ------------------ 3. Carga de Imagen y Previsualización ------------------
    st.header("🖼️ Cargar Imagen de Entrada")
    
    upload_col, preview_col = st.columns([3, 1])

    with upload_col:
        uploaded_file = st.file_uploader("Selecciona una imagen (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"], key="uploader_c2")
    
    with preview_col:
        st.markdown("<p style='font-size: 0.8em; margin-bottom: 0px;'>Vista Previa:</p>", unsafe_allow_html=True)
        if uploaded_file is not None:
            st.image(uploaded_file, width=100)
        else:
            st.markdown("<div style='height: 100px; border: 1px dashed #ccc; padding: 5px; text-align: center; line-height: 80px; color: #888;'>Sin imagen</div>", unsafe_allow_html=True)


    # Inicializar el estado de sesión para el control de procesamiento
    if 'processed_image_c2' not in st.session_state:
        st.session_state.processed_image_c2 = False

    
    # ------------------ 4. Botón de Procesamiento ------------------
    if st.button("Aplicar Filtros", type="primary"):
        if uploaded_file is not None:
            # Leer el archivo subido
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            
            # Decodificar la imagen en formato OpenCV (BGR)
            img_cv2 = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Convertir a escala de grises para el procesamiento de bordes
            img_gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
            
            # Guardar la imagen en escala de grises en el estado de la sesión
            st.session_state.image_input_gray = img_gray
            st.session_state.processed_image_c2 = True
        else:
            st.error("Por favor, sube una imagen primero.")
            st.session_state.processed_image_c2 = False


    # ------------------ 5. Pestañas de Resultados ------------------
    if st.session_state.processed_image_c2:
        img_input_gray = st.session_state.image_input_gray
        
        st.markdown("---")
        st.header("Resultados de la Detección de Bordes")

        # Define las tres pestañas
        tab_lap_canny, tab_laplacian, tab_canny = st.tabs(["Laplacian y Canny", "Laplacian", "Canny"])

        # Contenido de la Pestaña 1 (Combinada: Original, Laplacian, Canny)
        display_laplacian_canny_tab(tab_lap_canny, img_input_gray)

        # Contenido de la Pestaña 2 (Solo Laplacian)
        display_single_output_tab(tab_laplacian, img_input_gray, 'laplacian', "Filtro Laplaciano")

        # Contenido de la Pestaña 3 (Solo Canny)
        display_single_output_tab(tab_canny, img_input_gray, 'canny', "Detector de Bordes Canny")


run_capitulo2()
