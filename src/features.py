import numpy as np  # Численные операции.
from sklearn.base import BaseEstimator, TransformerMixin  # Совместимость со sklearn.
from src.schema import PAY, BILL, PAID  # Группы признаков.


class Features(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self  # No fitted statistics. / Здесь нет обучаемых статистик.

    def transform(self, X):
        frame = X.copy()  # Preserve input data. / Не меняем вход.
        frame["delay_max"] = frame[PAY].max(
            axis=1
        )  # Maximum payment delay. / Наибольшая задержка.
        frame["delay_count"] = (
            frame[PAY].gt(0).sum(axis=1)
        )  # Число месяцев с задержкой.
        frame["bill_mean"] = frame[BILL].mean(
            axis=1
        )  # Mean bill amount. / Средний счет.
        frame["paid_mean"] = frame[PAID].mean(
            axis=1
        )  # Mean payment amount. / Средний платеж.
        frame["utilization"] = frame["BILL_AMT1"] / frame["LIMIT_BAL"].replace(
            0, np.nan
        )  # Credit utilization. / Использование лимита.
        return frame  # Агрегированные признаки.
