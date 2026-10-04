from sklearn.compose import ColumnTransformer  # Выбор столбцов.
from sklearn.impute import SimpleImputer  # Imputer — заполнение пропусков.
from sklearn.pipeline import Pipeline  # Pipeline — цепочка обработки.
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)  # Кодирование и масштабирование.
from src.features import Features  # Feature engineering — создание признаков.
from src.schema import CATEGORIES  # Категории.


def numeric_columns(frame):
    return [c for c in frame.columns if c not in CATEGORIES]  # Все остальные числа.


def create_pipeline(classifier):
    numeric = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )  # Медианы учатся только на train.
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )  # Новые категории безопасны.
    preprocess = ColumnTransformer(
        [("numeric", numeric, numeric_columns), ("category", categorical, CATEGORIES)]
    )  # Общая обработка.
    return Pipeline(
        [
            ("features", Features()),
            ("preprocess", preprocess),
            ("classifier", classifier),
        ]
    )  # Та же цепочка для API.
