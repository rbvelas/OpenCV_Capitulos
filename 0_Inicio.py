import streamlit as st
from PIL import Image
import os

# =========================
# CONFIGURACIÓN DE LA PÁGINA
# =========================
st.set_page_config(
    page_title="Inicio | OpenCV 3.x con Python",
    page_icon="📚",
    layout="wide"
)

# =========================
# SECCIÓN 1: PORTADA Y TÍTULO
# =========================
PATH_PORTADA = "img/portada_libro.jpg"

st.markdown(
    """
    <style>
        .title-text {
            font-size: 2.4em; 
            font-weight: 800; 
            margin-bottom: 0px;
            color: #1E3A8A;
        }
        .author-text {
            font-size: 1.1em; 
            font-weight: 400;
            margin-top: 0px;
            color: #4B5563;
        }
        .body-text {
            font-size: 1.1em;
            line-height: 1.6;
            color: #374151;
        }
    </style>
    """,
    unsafe_allow_html=True
)

try:
    portada = Image.open(PATH_PORTADA)
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image(portada, width=200)
    with col2:
        st.markdown("<h1 class='title-text'>OpenCV 3.x with Python By Example</h1>", unsafe_allow_html=True)
        st.markdown("<p class='author-text'>Autor: Gabriel Garrido, Prateek Joshi</p>", unsafe_allow_html=True)
except FileNotFoundError:
    st.markdown("<h1 class='title-text'>OpenCV 3.x with Python By Example</h1>", unsafe_allow_html=True)
    st.markdown("<p class='author-text'>Autor: Gabriel Garrido, Prateek Joshi</p>", unsafe_allow_html=True)
    st.warning(f"No se encontró la imagen de portada en la ruta: {PATH_PORTADA}. "
               "Verifique la existencia de la carpeta 'img/' y el archivo correspondiente.")

# =========================
# SECCIÓN 2: DESCRIPCIÓN DE LA APP
# =========================
st.markdown(
    """
    <div class='body-text' style='margin-top: 20px;'>
        Esta aplicación interactiva ofrece una colección de ejercicios prácticos 
        inspirados en los capítulos del libro <b>“OpenCV 3.x with Python By Example”</b>. 
        El propósito es facilitar el aprendizaje del <b>procesamiento de imágenes</b> y la 
        <b>visión por computadora</b> a través de demostraciones visuales y manipulaciones en tiempo real.
        <br><br>
        Utilice el menú lateral para navegar entre los diferentes capítulos y explorar cada tema.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# =========================
# SECCIÓN 3: INFORMACIÓN DEL DESARROLLADOR
# =========================
PATH_LOGO_UNT = "img/UNT_logo.png"

try:
    logo_unt = Image.open(PATH_LOGO_UNT)
    col_dev, col_logo = st.columns([5, 1])
    with col_dev:
        st.markdown(
            """
            <p style='font-size: 0.95em; font-weight: 600; margin-bottom: 3px;'>
                Desarrollado por: <span style='color:#1E3A8A;'>Velasquez García, Ricardo Bernardo</span>
            </p>
            <p style='font-size: 0.9em; margin-top: 0px; margin-bottom: 2px; color:#4B5563;'>
                Escuela Profesional de Ingeniería de Sistemas
            </p>
            <p style='font-size: 0.9em; margin-top: 0px; color:#4B5563;'>
                Universidad Nacional de Trujillo
            </p>
            """,
            unsafe_allow_html=True
        )
    with col_logo:
        st.image(logo_unt, width=80)
except FileNotFoundError:
    st.caption("**Desarrollado por:** Velasquez García, Ricardo Bernardo")
    st.caption("Escuela Profesional de Ingeniería de Sistemas")
    st.caption("Universidad Nacional de Trujillo")
    st.warning(f"No se encontró el logo en la ruta: {PATH_LOGO_UNT}.")
