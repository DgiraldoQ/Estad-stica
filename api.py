from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import os
import google.generativeai as genai

# 1. Leer la API key (debe estar en Render)
genai_api_key = os.getenv("GOOGLE_API_KEY")

# 2. Configurar cliente Gemini
genai.configure(api_key=genai_api_key)

# 3. Cargar tu modelo entrenado .joblib
modelo = joblib.load("predict_decision_tree.joblib")

# 4. Inicializar FastAPI
app = FastAPI(title="API Calidad del Aire")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 5. Modelo pydantic
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

def generar_explicacion_llm(entrada, prediccion):
    # Construye prompt con los datos y el resultado
    prompt = (
        f"Actúa como experto en calidad del aire.\n"
        f"Estos son los valores AQI:\n"
        f"CO: {entrada[0]}, NO2: {entrada[1]}, SO2: {entrada[2]}, O3: {entrada[3]}, "
        f"PM2.5: {entrada[4]}, PM10: {entrada[5]}, AQI_TOTAL: {entrada[6]}.\n"
        f"El modelo predice una clasificación ({prediccion}).\n"
        "Explica al usuario qué significa este resultado para la salud pública y da 2 recomendaciones prácticas."
    )

    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt)
    return response.text.strip()

@app.post("/predict")
def predict(data: AirQualityInput):
    entrada = [
        data.CO_AQI,
        data.NO2_AQI,
        data.SO2_AQI,
        data.O3_AQI,
        data.PM25_AQI,
        data.PM10_AQI,
        data.AQI_TOTAL
    ]
    prediccion = int(modelo.predict([entrada])[0])
    explicacion = generar_explicacion_llm(entrada, prediccion)
    return {
        "clasificacion": prediccion,
        "explicacion_ia": explicacion
    }
