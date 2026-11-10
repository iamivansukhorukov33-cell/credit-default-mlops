from contextlib import asynccontextmanager  # Загрузка при старте.
import os  # Настройка пути модели.
import joblib  # Загрузка модели.
import pandas as pd  # Признаки в таблицу.
from fastapi import FastAPI  # REST API.
from pydantic import ConfigDict, Field, create_model  # Строгая схема запроса.
from src.schema import FEATURES, PAY, PAID  # Все 23 признака.

fields = {}  # Обязательные поля.
for column in FEATURES:
    bounds = (
        {"ge": 18, "le": 100}
        if column == "AGE"
        else (
            {"ge": -2, "le": 9}
            if column in PAY
            else ({"ge": 0} if column in PAID else {})
        )
    )  # Диапазоны.
    if column == "LIMIT_BAL":
        bounds = {"gt": 0, "le": 2000000}  # Лимит не нулевой.
    if column in ["SEX", "EDUCATION", "MARRIAGE"]:
        bounds = {
            "ge": 1,
            "le": {"SEX": 2, "EDUCATION": 4, "MARRIAGE": 3}[column],
        }  # Очищенные коды.
    kind = (
        int if column in ["AGE", "SEX", "EDUCATION", "MARRIAGE", *PAY] else float
    )  # Категории целочисленные.
    fields[column] = (
        kind,
        Field(..., allow_inf_nan=False, **bounds),
    )  # Reject NaN and infinity. / Никаких NaN/inf.
Client = create_model(
    "Client", __config__=ConfigDict(extra="forbid", strict=True), **fields
)  # Не допускаем лишние поля.


@asynccontextmanager
async def lifespan(app):
    path = os.getenv("MODEL_PATH", "models/pipeline.joblib")  # Можно изменить путь.
    app.state.model = joblib.load(
        path
    )  # Fail startup if model is missing. / Ошибка старта при отсутствии модели.
    yield  # Приложение работает.


app = FastAPI(title="Credit default API", lifespan=lifespan)  # Swagger на /docs.


@app.get("/health")
def health():
    return {"status": "ok"}  # Проверка работающего сервиса.


@app.post("/predict")
def predict(client: Client):
    frame = pd.DataFrame([client.model_dump()])[FEATURES]  # Порядок обучения.
    probability = float(
        app.state.model.predict_proba(frame)[0, 1]
    )  # Вероятность класса 1.
    return {
        "default_prediction": int(probability >= 0.5),
        "default_probability": probability,
    }  # Класс и вероятность.
