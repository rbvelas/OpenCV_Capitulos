import streamlit as st
import cv2
import numpy as np
import io

# Título de la Página
st.set_page_config(page_title="Capítulo 1", page_icon="1️⃣")


def transform_image(img_input, transform_type):
    """
    Aplica una transformación afín a la imagen de entrada.

    Args:
        img_input (np.array): Imagen de entrada en formato BGR (OpenCV).
        transform_type (str): Tipo de transformación a aplicar ('paralelogramo', 'espejo_h', 'espejo_v').

    Returns:
        np.array: Imagen transformada.
    """
    rows, cols = img_input.shape[:2]
    
    # Puntos de origen (esquina superior izquierda, superior derecha, inferior izquierda)
    src_points = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
    
    if transform_type == 'paralelogramo':
        # Mapea los puntos de origen a los nuevos puntos para crear el efecto de paralelogramo
        dst_points = np.float32([[0, 0], [int(0.6 * (cols - 1)), 0], [int(0.4 * (cols - 1)), rows - 1]])
        display_title = "Salida: Transformación a Paralelogramo (Sesgado)"
    
    elif transform_type == 'espejo_h':
        # Reflejo horizontal: Invierte los puntos X (superior derecha va a superior izquierda, viceversa)
        dst_points = np.float32([[cols - 1, 0], [0, 0], [cols - 1, rows - 1]])
        display_title = "Salida: Espejo Horizontal (Flip X)"
    
    elif transform_type == 'espejo_v':
        # Reflejo vertical: Invierte los puntos Y (superior izquierda va a inferior izquierda, viceversa)
        dst_points = np.float32([[0, rows - 1], [cols - 1, rows - 1], [0, 0]])
        display_title = "Salida: Espejo Vertical (Flip Y)"
        
    else:
        # Esto no debería ocurrir si se llama correctamente
        return img_input, "No se aplicó ninguna transformación"

    # Calcular la matriz de transformación afín
    affine_matrix = cv2.getAffineTransform(src_points, dst_points)
    
    # Aplicar la transformación
    img_output = cv2.warpAffine(img_input, affine_matrix, (cols, rows))
    
    return img_output, display_title


def display_transformation_tab(tab, img_input, transform_type):
    """Función auxiliar para mostrar el contenido de una pestaña de transformación."""
    with tab:
        if st.session_state.processed_image:
            img_output, display_title = transform_image(img_input, transform_type)
            
            # Usar columnas para mostrar Original vs Transformada
            col_orig, col_out = st.columns(2)
            
            with col_orig:
                st.caption("Entrada Original")
                # Aseguramos que la imagen se muestre en formato RGB para Streamlit
                st.image(cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB), use_container_width=True)
            with col_out:
                st.caption(display_title)
                st.image(cv2.cvtColor(img_output, cv2.COLOR_BGR2RGB), use_container_width=True)
        else:
            st.warning("Pulsa 'Aplicar Transformaciones' después de subir una imagen para ver los resultados.")


def run_capitulo1():
    
    # ------------------ 1. Títulos ------------------
    st.title("CAPÍTULO 1: Aplicando Transformaciones Geométricas a Imágenes")
    st.markdown("##### *Applying Geometric Transformations to Images*")

    st.markdown("---")

    # ------------------ 2. Subtítulo y Concepto ------------------
    st.subheader("Transformaciones Afines | **Affine Transformations**")
    st.info("Una transformación afín es una función que preserva líneas paralelas. Se utiliza para rotar, escalar, trasladar y sesgar imágenes.")
    
    # ------------------ 3. Carga de Imagen y Previsualización ------------------
    st.header("🖼️ Cargar Imagen de Entrada")
    
    upload_col, preview_col = st.columns([3, 1])

    with upload_col:
        uploaded_file = st.file_uploader("Selecciona una imagen (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"], key="uploader")
    
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
    if st.button("Aplicar Transformaciones", type="primary"):
        if uploaded_file is not None:
            # Lee el archivo subido como bytes
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            
            # Decodifica la imagen en formato OpenCV (BGR)
            img_cv2 = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Guarda la imagen procesada en el estado de la sesión
            st.session_state.image_input = img_cv2
            st.session_state.processed_image = True
        else:
            st.error("Por favor, sube una imagen primero.")
            st.session_state.processed_image = False


    # ------------------ 5. Pestañas de Resultados (Navegación Alternativa) ------------------
    if st.session_state.processed_image:
        img_input = st.session_state.image_input
        
        st.markdown("---")
        st.header("Resultados de las Transformaciones Afines")

        # Define las pestañas
        tab1, tab2, tab3 = st.tabs(["Paralelogramo", "Espejo Horizontal", "Espejo Vertical"])

        # Contenido de la Pestaña 1 (Paralelogramo)
        display_transformation_tab(tab1, img_input, 'paralelogramo')

        # Contenido de la Pestaña 2 (Espejo Horizontal)
        display_transformation_tab(tab2, img_input, 'espejo_h')

        # Contenido de la Pestaña 3 (Espejo Vertical)
        display_transformation_tab(tab3, img_input, 'espejo_v')


run_capitulo1()
