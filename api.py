from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import joblib
import os
import google.generativeai as genai  # SDK de Google Gemini

# Configuración de API Key para Google Gemini
genai_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=genai_api_key)

# Cargar modelo entrenado
modelo = joblib.load("modelo_calidad_aire.joblib")

app = FastAPI(title="API Calidad del Aire AI Autónoma")

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelo para entrada solo con los 6 contaminantes
class AirQualityInput(BaseModel):
    CO_AQI: float = Field(..., example=10.0)
    NO2_AQI: float = Field(..., example=20.0)
    SO2_AQI: float = Field(..., example=5.0)
    O3_AQI: float = Field(..., example=30.0)
    PM25_AQI: float = Field(..., alias="PM2.5_AQI", example=15.0)
    PM10_AQI: float = Field(..., example=25.0)

    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                "CO_AQI": 10.0,
                "NO2_AQI": 20.0,
                "SO2_AQI": 5.0,
                "O3_AQI": 30.0,
                "PM2.5_AQI": 15.0,
                "PM10_AQI": 25.0
            }
        }


@app.get("/")
def home():
    return {"mensaje": "API Calidad del Aire AI Autónoma funcionando"}


# Función para calcular AQI_TOTAL basado en entrada (puedes ajustar fórmula)
def calcular_aqi_total(valores):
    # Aquí un ejemplo simple: suma o máximo de los AQI individuales, o fórmula personalizada
    return sum(valores)  # Simple suma; ajusta si tienes otra fórmula


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


def generar_explicacion_llm(entrada, aqi_total, prediccion):
    prompt = (
        f"Soy un experto en calidad del aire.\n"
        f"Los niveles de contaminantes medidos son:\n"
        f"CO_AQI: {entrada[0]}, NO2_AQI: {entrada[1]}, SO2_AQI: {entrada[2]}, O3_AQI: {entrada[3]}, "
        f"PM2.5_AQI: {entrada[4]}, PM10_AQI: {entrada[5]}.\n"
        f"El AQI total calculado es: {aqi_total}.\n"
        f"El modelo predice la siguiente clasificación de calidad del aire: {prediccion}.\n"
        "Explica al usuario el significado de esta clasificación y da 2 recomendaciones de salud pública."
    )
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content(prompt)
    return response.text.strip()


@app.post("/predict")
def predict(data: AirQualityInput):
    # Obtener las 6 variables en el orden que esperaba el modelo
    entrada = [
        data.CO_AQI,
        data.NO2_AQI,
        data.SO2_AQI,
        data.O3_AQI,
        data.PM25_AQI,
        data.PM10_AQI,
    ]

    # Calcular AQI_TOTAL internamente
    aqi_total = calcular_aqi_total(entrada)

    # Crear la entrada completa para el modelo (7 parámetros)
    entrada_modelo = entrada + [aqi_total]

    prediccion = int(modelo.predict([entrada_modelo])[0])

    explicacion_clasificacion = interpretar_clasificacion(prediccion)
    explicacion_ia = generar_explicacion_llm(entrada, aqi_total, prediccion)

    return {
        "clasificacion": prediccion,
        "descripcion": explicacion_clasificacion,
        "explicacion_ia": explicacion_ia,
        "AQI_TOTAL_calculado": aqi_total
    }    
