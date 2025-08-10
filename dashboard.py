# dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import os

st.set_page_config(page_title="Calidad del Aire", layout="wide")

# API URL: primero intenta st.secrets (Streamlit Cloud), luego env var, luego local
def get_api_url():
    try:
        # En Streamlit Cloud pones en Settings -> Secrets: API_URL = "https://<tu-api>"
        return st.secrets["API_URL"]
    except Exception:
        return os.environ.get("API_URL", "http://127.0.0.1:8000")

API_URL = get_api_url().rstrip("/")

# Cargar datos
df = pd.read_csv("proyecto_normalizado.csv", sep=";")

st.title("🌍 Dashboard - Calidad del Aire")

st.sidebar.header("Predicción rápida")
CO_AQI = st.sidebar.number_input("CO_AQI", min_value=0.0, value=5.0)
NO2_AQI = st.sidebar.number_input("NO2_AQI", min_value=0.0, value=10.0)
SO2_AQI = st.sidebar.number_input("SO2_AQI", min_value=0.0, value=1.0)
O3_AQI = st.sidebar.number_input("O3_AQI", min_value=0.0, value=20.0)
PM25_AQI = st.sidebar.number_input("PM2.5_AQI", min_value=0.0, value=30.0)
PM10_AQI = st.sidebar.number_input("PM10_AQI", min_value=0.0, value=12.0)

if st.sidebar.button("Predecir (API)"):
    payload = {
        "CO_AQI": CO_AQI,
        "NO2_AQI": NO2_AQI,
        "SO2_AQI": SO2_AQI,
        "O3_AQI": O3_AQI,
        "PM25_AQI": PM25_AQI,
        "PM10_AQI": PM10_AQI
    }
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        if r.status_code == 200:
            st.sidebar.success(f"Clasificación predicha: {r.json()['clasificacion']}")
        else:
            st.sidebar.error(f"Error API: {r.status_code}")
    except Exception as e:
        st.sidebar.error(f"No se pudo conectar con la API: {e}")

st.subheader("Vista previa de datos")
st.dataframe(df.head())

st.subheader("Distribuciones")
cols = ["CO_AQI","NO2_AQI","SO2_AQI","O3_AQI","PM2.5_AQI","PM10_AQI","AQI_TOTAL"]
for c in cols:
    if c in df.columns:
        fig = px.histogram(df, x=c, nbins=30, title=c)
        st.plotly_chart(fig, use_container_width=True)