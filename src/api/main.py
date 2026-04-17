"""
API Básica usando FastAPI para servir el modelo entrenado.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.features.build_features import clean_data, create_features, encode_categoricals

# Inicializamos la app
app = FastAPI(title="API de Predicción de Precios de Vivienda (California)", version="1.0")

# Esquema de entrada: variables crudas tal como vienen del dataset
class HousingFeatures(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    ocean_proximity: str  # ej: "INLAND", "NEAR BAY", "<1H OCEAN", "NEAR OCEAN", "ISLAND"

# Bundle global: modelo + scaler + columnas esperadas
bundle = None

@app.on_event("startup")
def load_model():
    global bundle
    try:
        bundle = joblib.load("models/best_model.joblib")
        print("Modelo cargado correctamente.")
        print(f"Columnas esperadas: {bundle['feature_cols']}")
    except Exception as e:
        print(f"Advertencia: No se pudo cargar el modelo — {e}")

@app.get("/")
def home():
    return {"mensaje": "Bienvenido a la API del Proyecto Final de Ciencia de Datos"}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": bundle is not None}

@app.post("/predict")
def predict_price(features: HousingFeatures):
    """
    Recibe las variables crudas de un distrito, aplica el mismo
    preprocesamiento usado en entrenamiento y retorna la predicción.
    """
    if bundle is None:
        return {"error": "El modelo no se ha cargado. Ejecuta train_model.py primero."}

    model       = bundle['model']
    scaler      = bundle['scaler']
    feature_cols = bundle['feature_cols']

    # 1. Convertir input a DataFrame
    df = pd.DataFrame([features.model_dump()])

    # 2. Aplicar el mismo preprocesamiento que en entrenamiento
    df = clean_data(df)
    df = create_features(df)
    df = encode_categoricals(df)

    # 3. Alinear columnas con las del modelo (OHE puede generar columnas distintas)
    df = df.reindex(columns=feature_cols, fill_value=0)

    # 4. Escalar y predecir
    X_scaled = scaler.transform(df)
    prediction = float(model.predict(X_scaled)[0])

    return {"predicted_price": round(prediction, 2)}

# Instrucciones para correr la API localmente:
# En la terminal, desde la raíz del proyecto ejecuta:
# uvicorn src.api.main:app --reload