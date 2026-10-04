import json  # Отчет дрейфа.
from pathlib import Path  # Пути.
import numpy as np  # Гистограммы.
import pandas as pd  # Данные.
import joblib  # Модель.
import httpx  # Запросы к API.
from src.schema import FEATURES  # Схема.


def psi(reference, current):
    reference, current = np.asarray(reference), np.asarray(current)  # Массивы.
    if not len(reference) or not len(current):
        raise ValueError("Empty distribution")  # Нельзя оценить пустые данные.
    edges = np.unique(
        np.quantile(reference, np.linspace(0, 1, 11))
    )  # Границы только по train.
    edges = np.r_[-np.inf, edges[1:-1], np.inf]  # Новые крайние значения не теряются.
    if len(edges) == 2:
        edges = np.array(
            [-np.inf, float(reference[0]) + 0.5, np.inf]
        )  # Постоянный признак.
    a = np.histogram(reference, bins=edges)[0] / len(reference)  # Эталонные доли.
    b = np.histogram(current, bins=edges)[0] / len(current)  # Новые доли.
    a, b = np.clip(a, 1e-6, 1), np.clip(b, 1e-6, 1)  # Защита логарифма.
    return float(np.sum((b - a) * np.log(b / a)))  # Population Stability Index.


def monitor(url="http://localhost:8000"):
    reference = pd.read_csv("data/processed/train.csv")[FEATURES]  # Эталон train.
    current = pd.read_csv("data/processed/test.csv")[FEATURES].head(
        300
    )  # Имитация новых клиентов.
    model = joblib.load("models/pipeline.joblib")  # Вероятности эталона.
    probabilities = []  # Ответы сервиса.
    with httpx.Client(timeout=30) as client:
        for row in current.to_dict(orient="records"):
            response = client.post(
                f"{url}/predict", json=row
            )  # Реальное обращение API.
            response.raise_for_status()  # Не скрываем ошибку.
            probabilities.append(
                response.json()["default_probability"]
            )  # Собираем прогнозы.
    values = {
        c: psi(reference[c], current[c]) for c in ["AGE", "LIMIT_BAL", "PAY_0"]
    }  # Дрейф ключевых признаков.
    values["probability"] = psi(
        model.predict_proba(reference)[:, 1], probabilities
    )  # Дрейф прогнозов.
    report = {
        "psi": values,
        "alert": any(v >= 0.2 for v in values.values()),
        "rows": len(current),
    }  # Учебный порог.
    Path("reports/drift.json").write_text(json.dumps(report, indent=2))  # Отчет.
    print(report)  # Результат мониторинга.


if __name__ == "__main__":
    monitor()  # API должен быть запущен.
