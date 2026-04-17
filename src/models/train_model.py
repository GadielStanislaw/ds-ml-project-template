"""
Script de entrenamiento para el modelo final y evaluación básica.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
    train = pd.read_csv(processed_train_data_path)

    X_train = train.drop('median_house_value', axis=1)
    y_train = train['median_house_value']

    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)

    model = RandomForestRegressor(n_estimators=100, max_features='sqrt', random_state=42, n_jobs=-1)
    model.fit(X_train_s, y_train)

    bundle = {
        'model': model,
        'scaler': scaler,
        'feature_cols': list(X_train_s.columns),
    }
    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_save_path)
    print(f"Modelo guardado en: {model_save_path}")

def evaluate_model(model_path: str, processed_test_data_path: str):
    """
    INSTRUCCIONES:
    1. Carga el modelo guardado con joblib.load().
    2. Carga los datos de prueba preprocesados.
    3. Genera predicciones (y_pred) sobre los datos de prueba usando predict().
    4. Compara y_pred con las etiquetas reales calculando el RMSE y repórtalo en la terminal.
    """
    bundle = joblib.load(model_path)
    model = bundle['model']
    scaler = bundle['scaler']
    feature_cols = bundle['feature_cols']

    test = pd.read_csv(processed_test_data_path)
    X_test = test.drop('median_house_value', axis=1)
    y_test = test['median_house_value']

    X_test = X_test.reindex(columns=feature_cols, fill_value=0)
    X_test_s = scaler.transform(X_test)

    y_pred = model.predict(X_test_s)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    print(f"RMSE en Test Set: {rmse:,.0f} USD")
    print(f"MAE  en Test Set: {mae:,.0f} USD")
    print(f"R²   en Test Set: {r2:.4f}")

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[2]
    PROCESSED_TRAIN_PATH = str(ROOT / "data" / "processed" / "train_processed.csv")
    PROCESSED_TEST_PATH = str(ROOT / "data" / "processed" / "test_processed.csv")
    MODEL_OUTPUT_PATH = str(ROOT / "models" / "best_model.joblib")
    train_best_model(PROCESSED_TRAIN_PATH, MODEL_OUTPUT_PATH)
    evaluate_model(MODEL_OUTPUT_PATH, PROCESSED_TEST_PATH)
