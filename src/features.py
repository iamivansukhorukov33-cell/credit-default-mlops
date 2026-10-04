import numpy as np  # Численные операции.
from sklearn.base import BaseEstimator, TransformerMixin  # Совместимость со sklearn.
from src.schema import PAY, BILL, PAID  # Группы признаков.


class Features(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self  # Здесь нет обучаемых статистик.

    def transform(self, X):
        frame = X.copy()  # Не меняем вход.
        frame["delay_max"] = frame[PAY].max(axis=1)  # Наибольшая задержка.
        frame["delay_count"] = (
            frame[PAY].gt(0).sum(axis=1)
        )  # Число месяцев с задержкой.
        frame["bill_mean"] = frame[BILL].mean(axis=1)  # Средний счет.
        frame["paid_mean"] = frame[PAID].mean(axis=1)  # Средний платеж.
        frame["utilization"] = frame["BILL_AMT1"] / frame["LIMIT_BAL"].replace(
            0, np.nan
        )  # Использование лимита.
        return frame  # Агрегированные признаки.
