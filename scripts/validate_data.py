import pandas as pd  # Таблица.
from src.data import clean  # Очистка перед проверкой.
from src.validation import validate  # Контракт GX.

if __name__ == "__main__":
    validate(
        clean(pd.read_csv("data/raw/credit.csv")), "reports/validation.json"
    )  # Проверка реальных данных.
