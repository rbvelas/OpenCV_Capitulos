import streamlit as st
import cv2
import numpy as np
import io

# Título de la Página
st.set_page_config(page_title="Capítulo 3", page_icon="3️⃣")

# --- Funciones de Procesamiento ---

def process_invert_colors(img_cv2, x0, y0, x1, y1):
    """
    Simula la interacción del mouse invirtiendo los colores
    en una Región de Interés (ROI) definida.
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
    Adaptado de Codigo 2.
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
    img_small = cv2.resize(img, None, fx=1.0/ds_factor, fy=1.0/ds_factor, interpolation=cv2.INTER_AREA)
    
    # Apply bilateral filter the image multiple times 
    for i in range(num_repetitions): 
        img_small = cv2.bilateralFilter(img_small, ksize, sigma_color, sigma_space) 
    
    img_output = cv2.resize(img_small, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_LINEAR) 
    
    # Requerimos el mask de 3 canales para la operación bitwise_and
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Add the thick boundary lines to the image using 'AND' operator 
    dst = cv2.bitwise_and(img_output, img_output, mask=mask_bgr) 
    return dst


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
    st.info("Este capítulo explora cómo capturar datos de video (simulado con una foto de la cámara) y aplicar transformaciones complejas como la caricaturización.")

    st.header("📸 Capturar Frame de la Cámara")
    
    # Usamos st.camera_input para simular la captura de un frame
    camera_file = st.camera_input("Toma una foto para empezar a procesar:")

    if camera_file is None:
        st.info("Esperando la captura de una imagen de la cámara.")
        return # Sale de la función si no hay imagen
    
    # --- Procesamiento de la Imagen Capturada ---
    
    # Leer el archivo subido (byte array)
    file_bytes = np.asarray(bytearray(camera_file.read()), dtype=np.uint8)
    
    # Decodificar la imagen en formato OpenCV (BGR)
    img_input_cv2 = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Mostrar el frame capturado como imagen original
    st.image(img_input_cv2, channels="BGR", caption="Frame Original Capturado")

    st.markdown("---")
    st.header("Aplicar Procesamientos")

    # ------------------ 3. Configuración de Columnas de Salida ------------------
    
    col_invert, col_cartoon = st.columns(2)

    # --- COLUMNA IZQUIERDA: Inversión de Colores (Simulación de Mouse) ---
    with col_invert:
        st.subheader("1. Invertir Colores en Región (ROI)")
        st.caption("Simulación de interacción con mouse, ajustando la zona a invertir.")
        
        rows, cols = img_input_cv2.shape[:2]

        # Controles para definir la ROI (Región de Interés)
        st.markdown("**Definir Esquina Superior Izquierda (x0, y0):**")
        c1, c2 = st.columns(2)
        with c1:
            x0 = st.number_input("Coordenada X inicial (x0)", 0, cols, value=0, key="x0")
        with c2:
            y0 = st.number_input("Coordenada Y inicial (y0)", 0, rows, value=0, key="y0")

        st.markdown("**Definir Esquina Inferior Derecha (x1, y1):**")
        c3, c4 = st.columns(2)
        with c3:
            x1 = st.number_input("Coordenada X final (x1)", 0, cols, value=cols, key="x1")
        with c4:
            y1 = st.number_input("Coordenada Y final (y1)", 0, rows, value=rows, key="y1")

        # Procesar y mostrar la imagen invertida
        img_inverted = process_invert_colors(img_input_cv2, x0, y0, x1, y1)
        st.image(img_inverted, channels="BGR", caption="Resultado: Inversión de Colores en ROI")
        
        # Botón de Descarga
        get_image_download_link(img_inverted, "inverted_roi_image.png", "⬇️ Descargar Imagen Invertida")


    # --- COLUMNA DERECHA: Efecto Caricatura ---
    with col_cartoon:
        st.subheader("2. Efecto Caricatura")
        st.caption("Selecciona el modo de transformación.")
        
        # Selector de modo (Sketch o Color)
        mode = st.radio("Modo de Caricaturización", ["Caricatura a Color", "Sketch (Sin Color)"], horizontal=True)

        sketch_mode = (mode == "Sketch (Sin Color)")
        
        # Procesar la imagen con el efecto
        img_cartoon = cartoonize_image(img_input_cv2, sketch_mode=sketch_mode)
        st.image(img_cartoon, channels="BGR", caption=f"Resultado: {mode}")

        # Botón de Descarga
        get_image_download_link(img_cartoon, "cartoonized_image.png", "⬇️ Descargar Imagen Caricaturizada")


run_capitulo3()
