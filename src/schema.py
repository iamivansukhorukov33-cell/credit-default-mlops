TARGET = "default"  # Цель: дефолт в следующем месяце.
PAY = ["PAY_0", *[f"PAY_{i}" for i in range(2, 7)]]  # Шесть месяцев задержек.
BILL = [f"BILL_AMT{i}" for i in range(1, 7)]  # Суммы счетов.
PAID = [f"PAY_AMT{i}" for i in range(1, 7)]  # Суммы платежей.
CATEGORIES = ["SEX", "EDUCATION", "MARRIAGE", *PAY]  # Категориальные признаки.
FEATURES = [
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    *PAY,
    *BILL,
    *PAID,
]  # Без ID.
