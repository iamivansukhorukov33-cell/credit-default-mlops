from pathlib import Path  # Папки отчетов.
import matplotlib  # Серверный backend.

matplotlib.use("Agg")  # Без графического окна.
import matplotlib.pyplot as plt  # Визуализация.
import pandas as pd  # Анализ таблиц.
from src.schema import TARGET  # Название цели.


def eda():
    frame = pd.read_csv(
        "data/processed/train.csv"
    )  # EDA только по train: не подсматриваем test.
    Path("reports/eda").mkdir(parents=True, exist_ok=True)  # Папка графиков.
    frame.describe().to_csv("reports/eda/summary.csv")  # Описательная статистика.
    frame[TARGET].value_counts().plot.bar(
        title="Default class distribution (train)"
    )  # Дисбаланс классов.
    plt.savefig("reports/eda/classes.png", bbox_inches="tight")  # Сохраняем график.
    plt.close()  # Закрываем.
    frame.hist(
        column=["AGE", "LIMIT_BAL", "PAY_0", "BILL_AMT1"], bins=25, figsize=(10, 7)
    )  # Основные распределения.
    plt.tight_layout()  # Размещаем подписи.
    plt.savefig("reports/eda/distributions.png")  # Графики признаков.
    plt.close("all")  # Закрываем окна.
    correlations = (
        frame.drop(columns="ID").corr()[TARGET].drop(TARGET).sort_values()
    )  # Связь с целью без ID.
    correlations.to_csv(
        "reports/eda/target_correlations.csv"
    )  # Линейная связь, не причинность.
    correlations.tail(12).plot.barh(
        title="Correlation with default (train)", figsize=(8, 6)
    )  # Наиболее сильные положительные связи.
    plt.tight_layout()  # Подписи не обрезаются.
    plt.savefig("reports/eda/target_correlations.png")  # График для обсуждения.
    plt.close()  # Закрываем.
    frame.groupby("PAY_0")[TARGET].agg(["count", "mean"]).to_csv(
        "reports/eda/default_by_delay.csv"
    )  # Число клиентов и доля дефолтов по задержке.
    rate = frame[TARGET].mean()  # Доля дефолтов.
    text = f"# EDA обучающей выборки\n\nСтрок: {len(frame)}. Доля дефолта: {rate:.2%}. Пропусков: {int(frame.isna().sum().sum())}.\n\nID исключён из модели. Отрицательные счета сохранены. Коды EDUCATION 0/5/6 объединены с 4; MARRIAGE 0 с 3. Дисбаланс проверяем через ROC-AUC, precision, recall и F1; accuracy не является основным критерием. Выбросы не удаляем автоматически.\n"
    Path("reports/eda/analysis.md").write_text(text)  # Выводы, а не только графики.


if __name__ == "__main__":
    eda()  # Запуск анализа.
