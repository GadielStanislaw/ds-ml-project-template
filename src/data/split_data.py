"""
Script para dividir los datos en conjunto de entrenamiento y conjunto de prueba.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit

def split_and_save_data(raw_data_path: str, interim_data_path: str):
    """
    INSTRUCCIONES:
    1. Lee el archivo CSV descargado previamente en `raw_data_path` usando pandas.
    2. Separa los datos con `train_test_split()`. Te recomendamos un test_size=0.2 y random_state=42.
    3. (Opcional pero recomendado) Puedes usar `StratifiedShuffleSplit` basado en la variable
       del ingreso medio (median_income) para que la muestra sea representativa.
    4. Guarda los archivos resultantes (ej. train_set.csv y test_set.csv) en la carpeta `interim_data_path`.
    """
    housing = pd.read_csv(raw_data_path)

    # Crear categoría de ingreso para estratificación (5 bins) y evitar data leakage
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5]
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in split.split(housing, housing["income_cat"]):
        train_set = housing.loc[train_idx].drop("income_cat", axis=1)
        test_set = housing.loc[test_idx].drop("income_cat", axis=1)

    Path(interim_data_path).mkdir(parents=True, exist_ok=True)
    train_set.to_csv(Path(interim_data_path) / "train_set.csv", index=False)
    test_set.to_csv(Path(interim_data_path) / "test_set.csv", index=False)

    print(f"Train: {len(train_set)} filas | Test: {len(test_set)} filas")
    print(f"Guardados en: {interim_data_path}")

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[2]
    RAW_PATH = str(ROOT / "data" / "raw" / "housing" / "housing.csv")
    INTERIM_PATH = str(ROOT / "data" / "interim")
    split_and_save_data(RAW_PATH, INTERIM_PATH)