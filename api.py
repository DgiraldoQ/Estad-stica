# api.py
from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import joblib

# Cargar modelo
modelo = joblib.load("modelo_calidad_aire.joblib")

# Crear instancia de FastAPI
app = FastAPI(title="API Calidad del Aire")

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Datos de entrada
class AirQualityInput(BaseModel):
    CO_AQI: float
    NO2_AQI: float
    SO2_AQI: float
    O3_AQI: float
    PM2_5_AQI: float = Field(..., alias="PM2.5_AQI")  # alias para nombres con punto
    PM10_AQI: float
    AQI_TOTAL: float

    class Config:
        allow_population_by_field_name = True

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
        data.PM2_5_AQI,
        data.PM10_AQI,
        data.AQI_TOTAL
    ]]
    prediccion = modelo.predict(entrada)[0]
    return {"clasificacion": int(prediccion)}
