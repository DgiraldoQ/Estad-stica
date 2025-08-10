import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import os

st.set_page_config(page_title="Calidad del Aire", layout="wide")

# Obtener la URL de la API (puede venir de secrets o variable de entorno)
def get_api_url():
    try:
        return st.secrets["API_URL"]
    except Exception:
        return os.environ.get("API_URL", "http://127.0.0.1:8000")

API_URL = get_api_url().rstrip("/")

# Cargar dataset para visualizaciones
try:
    df = pd.read_csv("proyecto_normalizado.csv", sep=";")
except Exception:
    st.error("No se pudo cargar el archivo proyecto_normalizado.csv")
    st.stop()

st.title("🌍 Dashboard - Calidad del Aire")

# -------------------------------------
# Sección: Predicción rápida (sidebar)
# -------------------------------------
st.sidebar.header("Predicción rápida")

CO_AQI = st.sidebar.number_input("CO_AQI", min_value=0.0, value=5.0)
NO2_AQI = st.sidebar.number_input("NO2_AQI", min_value=0.0, value=10.0)
SO2_AQI = st.sidebar.number_input("SO2_AQI", min_value=0.0, value=1.0)
O3_AQI = st.sidebar.number_input("O3_AQI", min_value=0.0, value=20.0)
PM25_AQI = st.sidebar.number_input("PM2.5_AQI", min_value=0.0, value=30.0)
PM10_AQI = st.sidebar.number_input("PM10_AQI", min_value=0.0, value=12.0)

if st.sidebar.button("Predecir con IA (API)"):
    payload = {
        "CO_AQI": CO_AQI,
        "NO2_AQI": NO2_AQI,
        "SO2_AQI": SO2_AQI,
        "O3_AQI": O3_AQI,
        "PM2.5_AQI": PM25_AQI,  # alias que entiende la API
        "PM10_AQI": PM10_AQI
    }
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
        r.raise_for_status()
        data_resp = r.json()

        st.sidebar.success(f"Clasificación: {data_resp.get('clasificacion')}")
        st.sidebar.info(f"Descripción: {data_resp.get('descripcion')}")
        st.sidebar.write("**Explicación IA:**")
        st.sidebar.write(data_resp.get("explicacion_ia", ""))
        st.sidebar.write(f"AQI_TOTAL calculado: {data_resp.get('AQI_TOTAL_calculado')}")
    except Exception as e:
        st.sidebar.error(f"Error al conectar con la API: {e}")

# -------------------------------------
# Sección: Vista previa de datos
# -------------------------------------
st.subheader("📋 Vista previa de datos")
st.dataframe(df.head())

# -------------------------------------
# Sección: Distribuciones
# -------------------------------------
st.subheader("📊 Distribuciones de variables")
cols = ["CO_AQI", "NO2_AQI", "SO2_AQI", "O3_AQI", "PM2.5_AQI", "PM10_AQI", "AQI_TOTAL"]
for c in cols:
    if c in df.columns:
        fig = px.histogram(df, x=c, nbins=30, title=f"Distribución de {c}")
        st.plotly_chart(fig, use_container_width=True)
