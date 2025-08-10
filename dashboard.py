import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import os

st.set_page_config(page_title="Calidad del Aire", layout="wide")

def get_api_url():
    try:
        return st.secrets["API_URL"]
    except Exception:
        return os.environ.get("API_URL", "http://127.0.0.1:8000")

API_URL = get_api_url().rstrip("/")

try:
    df = pd.read_csv("proyecto_normalizado.csv", sep=";")
except Exception:
    st.error("No se pudo cargar el archivo proyecto_normalizado.csv")
    st.stop()

st.title("🌍 Dashboard - Calidad del Aire")

st.sidebar.header("Predicción rápida")
CO_AQI = st.sidebar.number_input("CO_AQI", min_value=0.0, value=5.0)
NO2_AQI = st.sidebar.number_input("NO2_AQI", min_value=0.0, value=10.0)
SO2_AQI = st.sidebar.number_input("SO2_AQI", min_value=0.0, value=1.0)
O3_AQI = st.sidebar.number_input("O3_AQI", min_value=0.0, value=20.0)
PM25_AQI = st.sidebar.number_input("PM2.5_AQI", min_value=0.0, value=30.0)
PM10_AQI = st.sidebar.number_input("PM10_AQI", min_value=0.0, value=12.0)
AQI_TOTAL = st.sidebar.number_input("AQI_TOTAL", min_value=0.0, value=50.0)  # <-- NUEVO

if st.sidebar.button("Predecir (API)"):
    payload = {
        "CO_AQI": CO_AQI,
        "NO2_AQI": NO2_AQI,
        "SO2_AQI": SO2_AQI,
        "O3_AQI": O3_AQI,
        "PM25_AQI": PM25_AQI,
        "PM10_AQI": PM10_AQI,
        "AQI_TOTAL": AQI_TOTAL  # <-- agregar
    }
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        r.raise_for_status()
        resultado = r.json().get("clasificacion", "N/A")
        mensaje = r.json().get("mensaje") or r.json().get("explicacion_ia", "")
        st.sidebar.success(f"Clasificación predicha: {resultado}")
        if mensaje:
            st.sidebar.info(mensaje)
    except Exception as e:
        st.sidebar.error(f"No se pudo conectar con la API: {e}")

st.subheader("Vista previa de datos")
st.dataframe(df.head())

st.subheader("Distribuciones")
cols = ["CO_AQI", "NO2_AQI", "SO2_AQI", "O3_AQI", "PM2.5_AQI", "PM10_AQI", "AQI_TOTAL"]
for c in cols:
    if c in df.columns:
        fig = px.histogram(df, x=c, nbins=30, title=f"Distribución de {c}")
        st.plotly_chart(fig, use_container_width=True)
