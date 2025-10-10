import streamlit as st
import cv2
import numpy as np
import io

# Título de la Página (debe estar ANTES de cualquier función)
st.set_page_config(page_title="Capítulo 6", page_icon="6️⃣", layout="wide")

# --- Funciones de Utilidad ---

def reset_results():
    """Resetea el estado de sesión para ocultar los resultados y forzar un nuevo cálculo."""
    if 'processed_image' in st.session_state:
        st.session_state.processed_image = False

# --- Funciones de Seam Carving ---

def overlay_vertical_seam(img, seam):
    """Dibuja la costura vertical sobre la imagen (línea verde)."""
    img_seam_overlay = np.copy(img)
    # Extraer las coordenadas (fila, columna) de la costura
    x_coords, y_coords = np.transpose([(i, int(j)) for i, j in enumerate(seam)])
    # Dibujar la costura en verde
    img_seam_overlay[x_coords, y_coords] = (0, 255, 0)
    return img_seam_overlay


def compute_energy_matrix(img):
    """Calcula la matriz de energía de la imagen usando Sobel."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    abs_sobel_x = cv2.convertScaleAbs(sobel_x)
    abs_sobel_y = cv2.convertScaleAbs(sobel_y)
    # Suma ponderada de los gradientes
    return cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)


def find_vertical_seam(img, energy):
    """Encuentra la costura vertical de menor energía usando programación dinámica."""
    rows, cols = img.shape[:2]
    seam = np.zeros(rows, dtype=int)
    
    # dist_to almacena el costo acumulado mínimo para llegar a cada píxel
    dist_to = np.zeros(img.shape[:2]) + float('inf')
    dist_to[0, :] = energy[0, :]
    
    # edge_to almacena el camino de dónde vino (1: izq, 0: centro, -1: der)
    edge_to = np.zeros(img.shape[:2], dtype=int)

    for row in range(rows - 1):
        for col in range(cols):
            current_dist = dist_to[row, col]
            
            # 1. Mover Izquierda-Abajo (col - 1)
            if col > 0:
                cost = current_dist + energy[row + 1, col - 1]
                if cost < dist_to[row + 1, col - 1]:
                    dist_to[row + 1, col - 1] = cost
                    edge_to[row + 1, col - 1] = 1

            # 2. Mover Centro-Abajo (col)
            cost = current_dist + energy[row + 1, col]
            if cost < dist_to[row + 1, col]:
                dist_to[row + 1, col] = cost
                edge_to[row + 1, col] = 0

            # 3. Mover Derecha-Abajo (col + 1)
            if col < cols - 1:
                cost = current_dist + energy[row + 1, col + 1]
                if cost < dist_to[row + 1, col + 1]:
                    dist_to[row + 1, col + 1] = cost
                    edge_to[row + 1, col + 1] = -1

    # Retracing the path (desde la última fila hacia arriba)
    seam[rows - 1] = np.argmin(dist_to[rows - 1, :])
    for i in (x for x in reversed(range(rows)) if x > 0):
        # seam[i-1] es la columna en la fila superior
        seam[i - 1] = int(seam[i] + edge_to[i, seam[i]])

    return seam


def add_vertical_seam(img, seam):
    """Añade una costura vertical a la imagen interpolando los valores de los vecinos."""
    rows, cols = img.shape[:2]
    img_extended = np.zeros((rows, cols + 1, 3), dtype=np.uint8)

    for row in range(rows):
        col_to_insert = seam[row]
        
        # Copiar píxeles a la izquierda de la inserción
        img_extended[row, :col_to_insert] = img[row, :col_to_insert]
        
        # Interpolación: promedio de los vecinos
        if col_to_insert == 0:
            interpolated_value = img[row, 0]
        elif col_to_insert == cols:
            interpolated_value = img[row, cols - 1]
        else:
            v1 = img[row, col_to_insert - 1].astype(np.int32)
            v2 = img[row, col_to_insert].astype(np.int32)
            interpolated_value = (v1 + v2) // 2
        
        img_extended[row, col_to_insert] = interpolated_value.astype(np.uint8)
        
        # Copiar píxeles a la derecha de la inserción (desplazados)
        img_extended[row, col_to_insert + 1:] = img[row, col_to_insert:]

    return img_extended


def remove_vertical_seam(img, seam):
    """Elimina una costura vertical de la imagen."""
    rows, cols = img.shape[:2]
    # Crear una máscara booleana para los píxeles que *no* están en la costura
    mask = np.ones((rows, cols), dtype=bool)
    for r in range(rows):
        mask[r, seam[r]] = False
    
    # Aplicar la máscara y reformar la imagen
    img = img[mask].reshape((rows, cols - 1, 3))
    return img


def expand_image_seam_carving(img_input, num_seams):
    """Expande la imagen añadiendo costuras verticales de baja energía (duplicación)."""
    img_for_seam_finding = np.copy(img_input)
    img_output = np.copy(img_input)
    
    # Seams encontradas en la imagen original para visualización
    all_seams_overlay = np.copy(img_input)
    
    # Almacenar las costuras encontradas para luego insertarlas en el orden correcto
    seams_to_add = []
    
    # 1. Encontrar y Eliminar temporalmente las costuras de menor energía
    for i in range(num_seams):
        energy = compute_energy_matrix(img_for_seam_finding)
        seam = find_vertical_seam(img_for_seam_finding, energy)
        
        # Guardar la costura antes de eliminarla para saber dónde duplicar
        seams_to_add.append(seam) 
        
        # Visualizar costura en la imagen original
        all_seams_overlay = overlay_vertical_seam(all_seams_overlay, seam)

        # Eliminar temporalmente para el siguiente cálculo de energía
        img_for_seam_finding = remove_vertical_seam(img_for_seam_finding, seam)
    
    # 2. Reconstruir la imagen añadiendo las costuras duplicadas
    for seam in reversed(seams_to_add):
        current_seam = np.copy(seam)
        img_output = add_vertical_seam(img_output, current_seam)

    return img_output, all_seams_overlay


def reduce_image_seam_carving(img_input, num_seams):
    """Reduce la imagen eliminando costuras verticales de baja energía."""
    img = np.copy(img_input)
    img_overlay_seam = np.copy(img_input)
    
    # 1. Encontrar y eliminar costuras
    for i in range(num_seams):
        energy = compute_energy_matrix(img)
        seam = find_vertical_seam(img, energy)
        
        # Visualizar costura en la imagen original
        img_overlay_seam = overlay_vertical_seam(img_overlay_seam, seam)
        
        # Eliminar la costura
        img = remove_vertical_seam(img, seam)
        
    return img, img_overlay_seam


def run_capitulo6():
    
    # ------------------ 1. Títulos ------------------
    st.title("CAPÍTULO 6: Tallado de Costuras")
    st.markdown("###### _Seam Carving_")

    st.markdown("---")

    # ------------------ 2. Subtítulo y Concepto ------------------
    st.subheader("Expandir y Reducir Imagen mediante Seam Carving")
    st.markdown("###### _Expand and Reduce Image by Seam Carving_")
    
    st.info("El **Seam Carving** es una técnica de redimensionamiento consciente del contenido que identifica y manipula las 'costuras' (caminos de píxeles de baja energía) para expandir o reducir imágenes sin distorsionar las áreas importantes. **Advertencia**: El procesamiento puede tomar varios segundos dependiendo del número de costuras seleccionadas.")

    # ✅ INICIALIZAR TODAS LAS VARIABLES DE SESSION_STATE
    if 'processed_image' not in st.session_state:
        st.session_state.processed_image = False
    if 'image_input' not in st.session_state:
        st.session_state.image_input = None
    if 'image_result' not in st.session_state:
        st.session_state.image_result = None
    if 'image_seams' not in st.session_state:
        st.session_state.image_seams = None
    if 'operation' not in st.session_state:
        st.session_state.operation = None

    # ------------------ 3. Carga de Imagen y Previsualización ------------------
    st.header("🖼️ Cargar Imagen de Entrada")

    upload_col, preview_col = st.columns([3, 1])

    with upload_col:
        uploaded_file = st.file_uploader(
            "Selecciona una imagen (PNG, JPG, JPEG)", 
            type=["png", "jpg", "jpeg"], 
            key="uploader_c6"
        )

    with preview_col:
        st.markdown("<p style='font-size: 0.8em; margin-bottom: 0px;'>Vista Previa:</p>", unsafe_allow_html=True)
        if uploaded_file is not None:
            st.image(uploaded_file, width=100)
        else:
            st.markdown("<div style='height: 100px; border: 1px dashed #ccc; padding: 5px; text-align: center; line-height: 80px; color: #888;'>Sin imagen</div>", unsafe_allow_html=True)

    # ------------------ 4. Selección de Operación ------------------
    st.header("⚙️ Configuración")
    
    operation = st.radio(
        "Selecciona la operación:",
        ["Expandir (Añadir costuras)", "Reducir (Eliminar costuras)"],
        horizontal=True,
        key='operation_select',
        on_change=reset_results
    )

    num_seams = st.slider("Número de costuras a procesar", min_value=1, max_value=100, value=20, step=1)

    # ------------------ 5. Botón de Procesamiento ------------------
    if st.button("Aplicar Seam Carving", type="primary"):
        if uploaded_file is not None:
            with st.spinner(f'Procesando {num_seams} costuras...'):
                # ✅ IMPORTANTE: Resetear el puntero del archivo
                uploaded_file.seek(0)
                
                # Leer bytes del archivo subido
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img_cv2 = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                if img_cv2 is not None:
                    # Opcional: Redimensionar si es muy grande
                    height, width = img_cv2.shape[:2]
                    max_dim = 500
                    if max(height, width) > max_dim:
                        scale = max_dim / max(height, width)
                        new_w, new_h = int(width * scale), int(height * scale)
                        img_cv2 = cv2.resize(img_cv2, (new_w, new_h), interpolation=cv2.INTER_AREA)

                    if operation == "Expandir (Añadir costuras)":
                        img_result, img_seams = expand_image_seam_carving(img_cv2, num_seams)
                        operation_text = "expandida"
                    else:
                        img_result, img_seams = reduce_image_seam_carving(img_cv2, num_seams)
                        operation_text = "reducida"
                    
                    # Almacenar resultados en el estado de sesión
                    st.session_state.image_input = img_cv2
                    st.session_state.image_result = img_result
                    st.session_state.image_seams = img_seams
                    st.session_state.operation = operation
                    st.session_state.processed_image = True
                    st.success(f'¡Procesamiento completado! La imagen fue {operation_text} con {num_seams} costuras.')
                else:
                    st.error("Error al cargar la imagen. Por favor, intenta con otra imagen.")
                    st.session_state.processed_image = False
        else:
            st.error("Por favor, sube una imagen primero.")
            st.session_state.processed_image = False

    # ------------------ 6. Mostrar Resultados ------------------
    if st.session_state.processed_image and st.session_state.image_result is not None:
        st.markdown("---")
        st.header("Resultados del Seam Carving")

        # Pestañas para diferentes visualizaciones
        tab1, tab2 = st.tabs(["Comparación Original vs Procesada", "Visualización de Costuras"])

        with tab1:
            col_orig, col_result = st.columns(2)
            
            # --- Columna Original ---
            with col_orig:
                st.caption("Imagen Original")
                st.image(cv2.cvtColor(st.session_state.image_input, cv2.COLOR_BGR2RGB), use_container_width=True)
                st.markdown(f"**Dimensiones:** {st.session_state.image_input.shape[1]} × {st.session_state.image_input.shape[0]} píxeles")
            
            # --- Columna Resultado ---
            with col_result:
                result_label = "Expandida" if st.session_state.operation and "Expandir" in st.session_state.operation else "Reducida"
                st.caption(f"Imagen {result_label}")
                st.image(cv2.cvtColor(st.session_state.image_result, cv2.COLOR_BGR2RGB), use_container_width=True)
                st.markdown(f"**Dimensiones:** {st.session_state.image_result.shape[1]} × {st.session_state.image_result.shape[0]} píxeles")
                
                # Botón de descarga
                is_success, buffer = cv2.imencode(".png", st.session_state.image_result)
                if is_success:
                    bio = io.BytesIO(buffer.tobytes())
                    st.download_button(
                        label="⬇️ Descargar Resultado",
                        data=bio.getvalue(),
                        file_name=f"seam_carving_{result_label.lower()}.png",
                        mime="image/png",
                        key="download_c6"
                    )

        with tab2:
            st.caption("Costuras Identificadas (en verde) para Remoción/Duplicación")
            st.image(cv2.cvtColor(st.session_state.image_seams, cv2.COLOR_BGR2RGB), use_container_width=True)
            operation_desc = "duplicaron para expandir" if st.session_state.operation and "Expandir" in st.session_state.operation else "eliminaron para reducir"
            st.info(f"Las líneas verdes muestran las costuras de menor energía que se identificaron y {operation_desc} la imagen.")


run_capitulo6()