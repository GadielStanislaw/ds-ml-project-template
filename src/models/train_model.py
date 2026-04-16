"""
Script de entrenamiento para el modelo final y evaluación básica.
"""

import pandas as pd
import joblib
from pathlib import Path

# IMPORTANTE: Se debe importar los algoritmos que quieran usar, por ejemplo:
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error

def train_best_model(processed_train_data_path: str, model_save_path: str):
    """
    INSTRUCCIONES:
    1. Carga los datos de entrenamiento procesados (que ya pasaron por `build_features.py`).
    2. Separa las características (X) de la etiqueta a predecir (y = 'median_house_value').
    3. Instancia tu mejor modelo encontrado después de la fase de experimentación y "fine tuning"
       (Por ejemplo: RandomForestRegressor con los mejores hiperparámetros).
    4. Entrena el modelo haciendo fit(X, y).
    5. Guarda el modelo entrenado en `model_save_path` (ej. 'models/best_model.pkl') usando joblib.dump().
    """
    pass

def evaluate_model(model_path: str, processed_test_data_path: str):
    """
    INSTRUCCIONES:
    1. Carga el modelo guardado con joblib.load().
    2. Carga los datos de prueba preprocesados.
    3. Genera predicciones (y_pred) sobre los datos de prueba usando predict().
    4. Compara y_pred con las etiquetas reales calculando el RMSE y repórtalo en la terminal.
    """
    pass

if __name__ == "__main__":
    # PROCESSED_TRAIN_PATH = "data/processed/train_processed.csv"
    # PROCESSED_TEST_PATH = "data/processed/test_processed.csv"
    # MODEL_OUTPUT_PATH = "models/best_model.pkl"
    # train_best_model(PROCESSED_TRAIN_PATH, MODEL_OUTPUT_PATH)
    # evaluate_model(MODEL_OUTPUT_PATH, PROCESSED_TEST_PATH)
    print("Script de entrenamiento final... (Falta el código!)")
