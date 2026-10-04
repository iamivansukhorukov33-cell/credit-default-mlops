import pandas as pd  # Чтение данных.
from src.validation import validate  # Реальная проверка GE.

if __name__ == "__main__":
    validate(
        pd.read_csv("data/processed/train.csv"), "reports/validation.json"
    )  # При аномалии код выхода ненулевой.
