from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib

# Cargar modelo
modelo = joblib.load("predict_decision_tree.joblib")

# Crear instancia de FastAPI
app = FastAPI(title="API Calidad del Aire")

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir todos los orígenes
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
    PM25_AQI: float    # ← Ya no usamos alias ni punto
    PM10_AQI: float
    AQI_TOTAL: float

@app.get("/")
def home():
    return {"mensaje": "API de Calidad del Aire funcionando"}

@app.post("/predict")
def predict(data: AirQualityInput):
    # Respetar el orden de características como en el entrenamiento del modelo
    entrada = [[
        data.CO_AQI,
        data.NO2_AQI,
        data.SO2_AQI,
        data.O3_AQI,
        data.PM25_AQI,
        data.PM10_AQI,
        data.AQI_TOTAL
    ]]
    prediccion = modelo.predict(entrada)[0]
    return {"clasificacion": int(prediccion)}
