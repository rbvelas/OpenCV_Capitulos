# pages/capitulo1.py
import streamlit as st

def run_capitulo1():
    st.title("📖 Capítulo 1: Introducción de Conceptos")
    st.markdown("""
    Este capítulo sienta las bases. ¡Aquí podemos poner texto, gráficos interactivos, o lo que necesites!
    """)
    st.subheader("Configuración")
    st.slider("Control de Ejemplo", 0, 100, 50)
    
run_capitulo1()