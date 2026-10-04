import pytest  # Ожидаемые ошибки.
from src.data import clean  # Очистка.
from src.validation import validate  # Great Expectations.
from src.schema import TARGET  # Цель.


def test_clean_does_not_mutate(frame):
    raw = frame.rename(
        columns={TARGET: "default payment next month"}
    )  # Имитируем CSV Kaggle.
    raw.loc[0, "EDUCATION"] = 5  # Неописанный код.
    cleaned = clean(raw)  # Нормализация.
    assert cleaned.loc[0, "EDUCATION"] == 4  # Объединение с «другое».
    assert raw.loc[0, "EDUCATION"] == 5  # Исходник не изменен.


def test_valid_data(frame):
    assert validate(frame).success  # Настоящий GX, не mock.


@pytest.mark.parametrize(
    "column,value",
    [
        ("AGE", 150),
        ("SEX", 3),
        ("LIMIT_BAL", 0),
        ("PAY_AMT1", -1),
        (TARGET, 2),
        ("AGE", None),
    ],
)
def test_anomaly_fails(frame, column, value):
    frame.loc[0, column] = value  # Вносим аномалию.
    with pytest.raises(ValueError, match="validation failed"):
        validate(frame)  # Должен остановить pipeline/CI.


def test_duplicate_client(frame):
    frame.loc[1, "ID"] = frame.loc[0, "ID"]  # Повтор клиента.
    with pytest.raises(ValueError, match="Duplicate"):
        clean(frame)  # Нельзя смешать одного клиента между train/test.
