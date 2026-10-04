from pathlib import Path  # Папка моделей.
import joblib  # Временная тестовая модель.
import pandas as pd  # Обучающие данные.
from sklearn.linear_model import (
    LogisticRegression,
)  # Быстрая модель только для smoke test.
from src.model import create_pipeline  # Настоящий preprocessing.
from src.schema import FEATURES, TARGET  # Схема.

if __name__ == "__main__":
    frame = pd.read_csv("data/processed/train.csv").head(
        1000
    )  # Маленькая выборка для проверки инфраструктуры.
    model = create_pipeline(LogisticRegression(max_iter=1000))  # Учебная модель.
    model.fit(frame[FEATURES], frame[TARGET])  # Обучение.
    Path("models").mkdir(exist_ok=True)  # Выход.
    joblib.dump(model, "models/demo.joblib")  # Не заменяет итоговую модель.
