import numpy as np  # Сравнение массивов.
import joblib  # Проверка сохранения.
from sklearn.linear_model import LogisticRegression  # Маленькая тестовая модель.
from src.model import create_pipeline  # Реальный pipeline.
from src.features import Features  # Агрегаты.
from src.schema import FEATURES, TARGET  # Контракт.
from src.train import metrics  # Подсчет качества.
from src.monitor import psi  # Дрейф.


def test_features(frame):
    frame.loc[0, "PAY_0"] = 2  # Один месяц задержки.
    transformed = Features().fit_transform(frame[FEATURES])  # Преобразование.
    assert transformed.loc[0, "delay_max"] == 2  # Максимум.
    assert transformed.loc[0, "delay_count"] == 1  # Счетчик.
    assert "delay_max" not in frame  # Нет изменения исходника.


def test_pipeline_roundtrip(frame, tmp_path):
    model = create_pipeline(LogisticRegression(max_iter=1000))  # Полная цепочка.
    model.fit(frame[FEATURES], frame[TARGET])  # Обучение.
    path = tmp_path / "model.joblib"  # Временный файл.
    joblib.dump(model, path)  # Сохранение.
    assert np.allclose(
        model.predict_proba(frame[FEATURES]),
        joblib.load(path).predict_proba(frame[FEATURES]),
    )  # API получает ту же модель.


def test_metrics():
    result = metrics(
        [0, 1, 0, 1], np.array([0.1, 0.8, 0.2, 0.9])
    )  # Идеальные предсказания.
    assert all(value == 1 for value in result.values())  # Все метрики равны 1.


def test_psi():
    reference = np.arange(100)  # Эталон.
    assert psi(reference, reference) == 0  # Нет дрейфа.
    assert psi(reference, reference + 200) > 0.2  # Сильный сдвиг.


def test_psi_constant_shifts():
    reference = np.ones(100)
    assert psi(reference, reference) == 0
    assert psi(reference, reference - 1) > 0.2
    assert psi(reference, reference + 1) > 0.2


def test_psi_invalid():
    import pytest

    for current in ([], [np.nan], [np.inf], [[1, 2]]):
        with pytest.raises(ValueError):
            psi([1, 2], current)
