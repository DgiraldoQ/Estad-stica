from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib

# Cargar tu modelo entrenado (.joblib)
modelo = joblib.load("predict_decision_tree.joblib")

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

# Esquema de datos esperado
class AirQualityInput(BaseModel):
    CO_AQI: float
    NO2_AQI: float
    SO2_AQI: float
    O3_AQI: float
    PM25_AQI: float
    PM10_AQI: float
    AQI_TOTAL: float

@app.get("/")
def home():
    return {"mensaje": "API de Calidad del Aire funcionando"}

# Función para describir la clasificación devuelta por el modelo
def interpretar_clasificacion(clasificacion: int):
    explicaciones = {
        0: "Buena: Calidad del aire satisfactoria, sin riesgos.",
        1: "Moderada: Aceptable, riesgos leves para personas sensibles.",
        2: "Dañina para grupos sensibles: Riesgo para personas con afecciones respiratorias o cardíacas.",
        3: "Dañina: Puede afectar la salud de toda la población.",
        4: "Muy dañina: Alerta sanitaria, efectos graves para todos.",
        5: "Peligrosa: Emergencia, evite salir al aire libre."
    }
    return explicaciones.get(clasificacion, "Clasificación desconocida")

@app.post("/predict")
def predict(data: AirQualityInput):
    entrada = [[
        data.CO_AQI,
        data.NO2_AQI,
        data.SO2_AQI,
        data.O3_AQI,
        data.PM25_AQI,
        data.PM10_AQI,
        data.AQI_TOTAL
    ]]
    prediccion = int(modelo.predict(entrada)[0])
    explicacion = interpretar_clasificacion(prediccion)
    return {
        "clasificacion": prediccion,
        "mensaje": explicacion
    }
