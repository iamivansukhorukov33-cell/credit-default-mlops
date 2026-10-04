import joblib  # Сохраняем временную модель.
from fastapi.testclient import TestClient  # Запросы без сети.
from sklearn.linear_model import LogisticRegression  # Тестовое обучение.
from src.api import app  # Приложение.
from src.model import create_pipeline  # Настоящий pipeline.
from src.schema import FEATURES, TARGET  # Схема.


def test_api(frame, tmp_path, monkeypatch):
    model = create_pipeline(LogisticRegression(max_iter=1000)).fit(
        frame[FEATURES], frame[TARGET]
    )  # Тестовая модель.
    path = tmp_path / "pipeline.joblib"  # Временный артефакт.
    joblib.dump(model, path)  # Сохраняем.
    monkeypatch.setenv("MODEL_PATH", str(path))  # Lifespan загрузит тестовую модель.
    payload = frame[FEATURES].iloc[0].to_dict()  # Все признаки.
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)  # Прогноз.
        assert response.status_code == 200  # Успешный ответ.
        assert (
            0 <= response.json()["default_probability"] <= 1
        )  # Допустимая вероятность.
        assert response.json()["default_prediction"] in [0, 1]  # Допустимый класс.
        assert client.get("/health").status_code == 200  # Healthcheck.
        assert (
            client.post("/predict", json={"AGE": 30}).status_code == 422
        )  # Неполный запрос отвергается.
        payload["AGE"] = 150  # Аномалия.
        assert (
            client.post("/predict", json=payload).status_code == 422
        )  # Проверка диапазона.
