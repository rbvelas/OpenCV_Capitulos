import streamlit as st
import pandas as pd

# --- Definición de Páginas/Capítulos ---

def pagina_principal():
    """Muestra el contenido del Capítulo 1."""
    st.title("📚 Capítulo 1: Introducción y Página Principal")
    st.write("¡Bienvenido! Esta es la introducción de la aplicación. Usa el menú lateral para navegar a los Capítulos.")
    st.info("💡 **Consejo:** Aquí puedes describir el propósito de la serie de Capítulos.")

def cargar_datos():
    """Muestra el contenido del Capítulo 2."""
    st.title("📂 Capítulo 2: Carga y Procesamiento de Datos")
    st.write("Aquí podrás subir tus archivos para el análisis.")
    # Ejemplo de funcionalidad: cargador de archivos simulado
    uploaded_file = st.file_uploader("Sube un archivo CSV o Excel", type=["csv", "xlsx"])
    if uploaded_file is not None:
        st.success(f"Archivo '{uploaded_file.name}' cargado con éxito.")
        # Aquí iría la lógica real para leer y procesar el archivo.

def visualizar_datos():
    """Muestra el contenido del Capítulo 3."""
    st.title("📊 Capítulo 3: Visualización de Datos Inicial")
    st.write("Esta sección contendría gráficos e información extraída de los datos cargados.")
    # Ejemplo de visualización simulada (una tabla simple)
    data = {'col1': [1, 2, 3], 'col2': [10, 20, 30]}
    df = pd.DataFrame(data)
    st.subheader("Datos de Ejemplo")
    st.dataframe(df)

def capitulo_generico(numero):
    """Muestra el contenido para el resto de los Capítulos (4 al 11)."""
    st.title(f"📖 Capítulo {numero}: Contenido Detallado")
    st.write(f"Este es el contenido del **Capítulo {numero}**. Aquí iría la lógica específica, explicaciones o herramientas interactivas correspondientes a este capítulo.")
    st.balloons()


# --- Lógica Principal y Navegación ---

# 1. Crear la lista de Capítulos del 1 al 11
# Usamos una comprensión de lista para generar nombres legibles: "Capítulo 1", "Capítulo 2", etc.
opciones_capitulos = [f"Capítulo {i}" for i in range(1, 12)]

# Título del Sidebar
st.sidebar.title("📚 Módulos y Capítulos")

# Usa st.sidebar.radio para las opciones de Capítulos.
pagina_seleccionada = st.sidebar.radio(
    "Selecciona un Capítulo:",
    opciones_capitulos # Usamos la lista generada
)

# 2. Muestra el contenido de la página/capítulo seleccionado
# Primero verificamos los capítulos específicos (1, 2, 3) que tienen funciones definidas.
if pagina_seleccionada == "Capítulo 1":
    pagina_principal()
elif pagina_seleccionada == "Capítulo 2":
    cargar_datos()
elif pagina_seleccionada == "Capítulo 3":
    visualizar_datos()
# Para el resto de los capítulos (4 al 11), usamos la función genérica.
else:
    # Extraemos el número del string, por ejemplo, de "Capítulo 5" obtenemos el 5.
    try:
        numero_capitulo = int(pagina_seleccionada.split()[-1])
        capitulo_generico(numero_capitulo)
    except ValueError:
        st.error("Error al determinar el número de capítulo.")