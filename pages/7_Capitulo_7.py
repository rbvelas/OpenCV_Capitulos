import streamlit as st
import cv2
import numpy as np

# Título de la Página
st.set_page_config(page_title="Capítulo 7", page_icon="7️⃣")


def apply_watershed_segmentation(img_input, morph_iterations=2, dist_threshold=0.5):
    """
    Aplica el algoritmo Watershed para segmentar objetos en la imagen.
    
    Args:
        img_input (np.array): Imagen de entrada en formato BGR.
        morph_iterations (int): Número de iteraciones para operaciones morfológicas.
        dist_threshold (float): Umbral para la transformada de distancia (0.0-1.0).
    
    Returns:
        tuple: Diccionario con todas las imágenes intermedias y resultado final
    """
    # Convertir a escala de grises
    gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
    
    # Aplicar desenfoque gaussiano para reducir ruido
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Aplicar umbralización con Otsu
    ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Eliminación de ruido mediante apertura morfológica
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=morph_iterations)
    
    # Área de fondo seguro mediante dilatación
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # Encontrar área de primer plano seguro mediante transformada de distancia
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, dist_threshold * dist_transform.max(), 255, 0)
    
    # Encontrar región desconocida
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Etiquetado de marcadores
    ret, markers = cv2.connectedComponents(sure_fg)
    
    # Añadir 1 a todas las etiquetas para que el fondo seguro sea 1, no 0
    markers = markers + 1
    
    # Marcar la región desconocida con cero
    markers[unknown == 255] = 0
    
    # Aplicar el algoritmo Watershed
    img_result = np.copy(img_input)
    markers_watershed = cv2.watershed(img_result, markers)
    
    # Crear imagen con bordes marcados en rojo sobre la original
    img_result[markers_watershed == -1] = [0, 0, 255]
    
    # Crear imagen con segmentación coloreada con colores más vivos
    img_colored = np.zeros_like(img_input)
    
    # Generar paleta de colores distintos
    np.random.seed(42)  # Para consistencia
    num_labels = markers_watershed.max() + 1
    colors = np.random.randint(50, 255, size=(num_labels, 3))
    
    for label in range(2, num_labels):
        img_colored[markers_watershed == label] = colors[label]
    
    # Marcar bordes en blanco en la imagen coloreada
    img_colored[markers_watershed == -1] = [255, 255, 255]
    
    # Crear overlay semi-transparente sobre la imagen original
    img_overlay = img_input.copy().astype(np.float32)
    alpha = 0.6
    for label in range(2, num_labels):
        mask = markers_watershed == label
        color_overlay = np.zeros_like(img_input, dtype=np.float32)
        color_overlay[mask] = colors[label]
        img_overlay[mask] = img_input[mask] * (1 - alpha) + color_overlay[mask] * alpha
    
    img_overlay = img_overlay.astype(np.uint8)
    img_overlay[markers_watershed == -1] = [0, 0, 255]
    
    # Retornar todas las imágenes intermedias
    results = {
        'threshold': thresh,
        'opening': opening,
        'sure_bg': sure_bg,
        'dist_transform': dist_transform,
        'sure_fg': sure_fg,
        'unknown': unknown,
        'markers': markers_watershed,
        'result': img_result,
        'colored': img_colored,
        'overlay': img_overlay,
        'num_objects': num_labels - 2  # Excluir fondo y borde
    }
    
    return results


def run_capitulo7():
    # ------------------ 1. Títulos ------------------
    st.title("CAPÍTULO 7: Detección de Formas y Segmentación de Imágenes")
    st.markdown("##### *Detecting Shapes and Segmenting an Image*")

    st.markdown("---")

    # ------------------ 2. Subtítulo y Concepto ------------------
    st.subheader("Algoritmo Watershed | **Watershed Algorithm**")
    st.info("El algoritmo Watershed es una técnica de segmentación de imágenes que trata la imagen como un mapa topográfico. Identifica y separa objetos conectados mediante el análisis de 'cuencas' y 'crestas' en el gradiente de la imagen.")

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

    # ------------------ 4. Parámetros de Configuración ------------------
    st.header("⚙️ Parámetros de Segmentación")
    
    col_param1, col_param2 = st.columns(2)
    
    with col_param1:
        morph_iterations = st.slider(
            "Iteraciones de Apertura Morfológica",
            min_value=1,
            max_value=10,
            value=2,
            help="Controla la eliminación de ruido. Valores más altos eliminan más ruido pero pueden perder detalles pequeños."
        )
    
    with col_param2:
        dist_threshold = st.slider(
            "Umbral de Distancia (%)",
            min_value=10,
            max_value=90,
            value=50,
            help="Define qué tan seguros deben estar los píxeles para ser considerados objetos. Valores bajos detectan más objetos, valores altos son más conservadores."
        ) / 100.0

    # Inicializar el estado de sesión
    if 'processed_image' not in st.session_state:
        st.session_state.processed_image = False

    # ------------------ 5. Botón de Procesamiento ------------------
    if st.button("Aplicar Segmentación Watershed", type="primary"):
        if uploaded_file is not None:
            with st.spinner('Procesando segmentación...'):
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img_cv2 = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                results = apply_watershed_segmentation(img_cv2, morph_iterations, dist_threshold)
                
                st.session_state.image_input = img_cv2
                st.session_state.results = results
                st.session_state.processed_image = True
                st.success(f'¡Segmentación completada! Se detectaron {results["num_objects"]} objetos.')
        else:
            st.error("Por favor, sube una imagen primero.")
            st.session_state.processed_image = False

    # ------------------ 6. Mostrar Resultados ------------------
    if st.session_state.processed_image:
        st.markdown("---")
        st.header("Resultados de la Segmentación Watershed")
        
        results = st.session_state.results
        
        # Mostrar métrica destacada
        col_metric1, col_metric2, col_metric3 = st.columns(3)
        with col_metric1:
            st.metric("Objetos Detectados", results["num_objects"])
        with col_metric2:
            st.metric("Ancho Original", f"{st.session_state.image_input.shape[1]} px")
        with col_metric3:
            st.metric("Alto Original", f"{st.session_state.image_input.shape[0]} px")

        # Pestañas para diferentes visualizaciones
        tab1, tab2, tab3 = st.tabs(["Resultado Final", "Proceso Paso a Paso", "Análisis de Distancia"])

        with tab1:
            st.subheader("Visualizaciones de Segmentación")
            
            # Primera fila: Original y con bordes
            col1, col2 = st.columns(2)
            with col1:
                st.caption("Imagen Original")
                st.image(cv2.cvtColor(st.session_state.image_input, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            with col2:
                st.caption("Bordes de Segmentación (rojo)")
                st.image(cv2.cvtColor(results['result'], cv2.COLOR_BGR2RGB), use_container_width=True)
            
            # Segunda fila: Regiones coloreadas y overlay
            col3, col4 = st.columns(2)
            with col3:
                st.caption("Regiones Segmentadas")
                st.image(cv2.cvtColor(results['colored'], cv2.COLOR_BGR2RGB), use_container_width=True)
            
            with col4:
                st.caption("Overlay Semi-transparente")
                st.image(cv2.cvtColor(results['overlay'], cv2.COLOR_BGR2RGB), use_container_width=True)
            
            st.info("🎯 El algoritmo Watershed identifica y separa objetos automáticamente. Cada color representa un objeto diferente detectado en la imagen.")

        with tab2:
            st.subheader("Etapas del Procesamiento")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.caption("1. Umbralización con Otsu")
                st.image(results['threshold'], use_container_width=True)
                
                st.caption("3. Fondo Seguro (Sure Background)")
                st.image(results['sure_bg'], use_container_width=True)
                
                st.caption("5. Región Desconocida")
                st.image(results['unknown'], use_container_width=True)
            
            with col2:
                st.caption("2. Apertura Morfológica")
                st.image(results['opening'], use_container_width=True)
                
                st.caption("4. Primer Plano Seguro (Sure Foreground)")
                st.image(results['sure_fg'], use_container_width=True)
                
                st.caption("6. Resultado Final")
                st.image(cv2.cvtColor(results['result'], cv2.COLOR_BGR2RGB), use_container_width=True)
            
            st.markdown("""
            **Explicación del proceso:**
            1. **Umbralización:** Separación inicial de objetos del fondo
            2. **Apertura Morfológica:** Eliminación de ruido
            3. **Fondo Seguro:** Área que definitivamente es fondo
            4. **Primer Plano Seguro:** Área que definitivamente son objetos
            5. **Región Desconocida:** Área entre fondo y primer plano que necesita ser clasificada
            6. **Watershed:** Asignación de píxeles desconocidos y marcado de fronteras
            """)

        with tab3:
            st.subheader("Transformada de Distancia")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.caption("Transformada de Distancia")
                # Normalizar para visualización
                dist_normalized = cv2.normalize(results['dist_transform'], None, 0, 255, cv2.NORM_MINMAX)
                st.image(dist_normalized.astype(np.uint8), use_container_width=True)
            
            with col2:
                st.caption("Primer Plano Seguro")
                st.image(results['sure_fg'], use_container_width=True)
            
            st.info("La Transformada de Distancia calcula la distancia de cada píxel al píxel de fondo más cercano. Los píxeles más brillantes están más lejos del fondo, ayudando a identificar los centros de los objetos.")


run_capitulo7()