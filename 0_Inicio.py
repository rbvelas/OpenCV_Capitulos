# app.py (Tu archivo principal, fuera de la carpeta pages)
import streamlit as st

def main_page():
    st.set_page_config(
        page_title="App de Capítulos",
        page_icon="📚",
        layout="wide"
    )
    st.title("📚 Módulos y Capítulos")
    st.header("¡Bienvenido al índice!")
    st.write("Usa el **menú de la izquierda** para navegar a través de los 11 Capítulos.")
    st.info("💡 **Streamlit Tip:** Los Capítulos (páginas) están definidos en la carpeta `pages/`.")

if __name__ == "__main__":
    main_page()

# ¡No necesitas código de navegación! Streamlit lo hace automáticamente.