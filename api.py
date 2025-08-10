# api.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Cargar modelo
modelo = joblib.load("modelo_calidad_aire.joblib")

# Crear instancia de FastAPI
app = FastAPI(title="API Calidad del Aire")

# Clase para los datos de entrada
class AirQualityInput(BaseModel):
    CO_AQI: float
    NO2_AQI: float
    SO2_AQI: float
    O3_AQI: float
    PM25_AQI: float
    PM10_AQI: float

@app.get("/")
def home():
    return {"mensaje": "API de Calidad del Aire funcionando"}

@app.post("/predict")
def predict(data: AirQualityInput):
    entrada = [[
        data.CO_AQI,
        data.NO2_AQI,
        data.SO2_AQI,
        data.O3_AQI,
        data.PM25_AQI,
        data.PM10_AQI
    ]]
    prediccion = modelo.predict(entrada)[0]
    return {"clasificacion": int(prediccion)}