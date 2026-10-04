import pandas as pd  # Fixture — подготовленные тестовые данные.
import pytest  # Тестовая инфраструктура.
from src.schema import FEATURES, TARGET  # Общая схема.


@pytest.fixture
def frame():
    rows = []  # Искусственные клиенты, не оценка качества модели.
    for i in range(40):
        row = {c: 0 for c in FEATURES}  # Начальные признаки.
        row.update(
            ID=i + 1,
            LIMIT_BAL=10000 + i * 1000,
            AGE=21 + i,
            SEX=1 + i % 2,
            EDUCATION=1 + i % 4,
            MARRIAGE=1 + i % 3,
        )  # Допустимые значения.
        row[TARGET] = i % 2  # Два класса.
        rows.append(row)  # Добавляем клиента.
    return pd.DataFrame(rows)  # Таблица для тестов.
