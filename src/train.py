import json  # Метрики и сравнение.
from pathlib import Path  # Пути артефактов.
import joblib  # Сохранение pipeline.
import matplotlib  # Графики без окна.

matplotlib.use("Agg")  # Серверный режим.
import matplotlib.pyplot as plt  # ROC-кривая.
import mlflow  # Tracking — журнал экспериментов.
import mlflow.sklearn  # Сохранение sklearn в MLflow.
from mlflow.models import infer_signature  # Схема модели.
import pandas as pd  # Чтение выборок.
from sklearn.ensemble import GradientBoostingClassifier  # Бустинг деревьев.
from sklearn.linear_model import LogisticRegression  # Логистическая регрессия.
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    RocCurveDisplay,
)  # Метрики.
from sklearn.model_selection import GridSearchCV, StratifiedKFold  # Подбор по CV.
from src.model import create_pipeline  # Полная цепочка.
from src.schema import FEATURES, TARGET  # Без цели и ID.
from src.validation import validate  # Защита от испорченных данных.


def metrics(y, probability):
    labels = (
        probability >= 0.5
    )  # Fixed decision threshold. / Фиксированный порог; test не подбирает его.
    return {
        "roc_auc": float(roc_auc_score(y, probability)),
        "precision": float(precision_score(y, labels, zero_division=0)),
        "recall": float(recall_score(y, labels)),
        "f1": float(f1_score(y, labels)),
    }  # Четыре метрики.


def train():
    training = pd.read_csv(
        "data/processed/train.csv"
    )  # Только обучающие данные для поиска.
    validate(training)  # Проверяем вход.
    X, y = training[FEATURES], training[TARGET]  # Разделяем признаки и цель.
    mlflow.set_tracking_uri("sqlite:///mlflow.db")  # Локальная база tracking.
    mlflow.set_experiment("credit-default")  # Группа экспериментов.
    cv = StratifiedKFold(
        n_splits=3, shuffle=True, random_state=42
    )  # Три сбалансированных разбиения.
    candidates = [
        ("lr_01", LogisticRegression(max_iter=1500), {"classifier__C": [0.1]}),
        ("lr_1", LogisticRegression(max_iter=1500), {"classifier__C": [1.0]}),
        (
            "lr_balanced",
            LogisticRegression(max_iter=1500, class_weight="balanced"),
            {"classifier__C": [1.0]},
        ),
        (
            "gb_depth1",
            GradientBoostingClassifier(random_state=42),
            {"classifier__max_depth": [1], "classifier__n_estimators": [60, 100]},
        ),
        (
            "gb_depth2",
            GradientBoostingClassifier(random_state=42),
            {"classifier__max_depth": [2], "classifier__n_estimators": [60, 100]},
        ),
    ]  # Пять самостоятельных экспериментов, семь конфигураций.
    results, best_model, best_score = [], None, -1.0  # Сравниваем только CV ROC-AUC.
    Path("models").mkdir(exist_ok=True)  # Выход модели.
    Path("reports").mkdir(exist_ok=True)  # Отчеты.
    for name, classifier, grid in candidates:
        with mlflow.start_run(run_name=name) as run:
            search = GridSearchCV(
                create_pipeline(classifier), grid, scoring="roc_auc", cv=cv, n_jobs=1
            )  # Fit preprocessing inside CV. / Преобработка внутри CV.
            search.fit(X, y)  # Keep test data unseen. / Обучение без test.
            score = float(search.best_score_)  # Средний ROC-AUC.
            mlflow.log_params(
                {
                    "algorithm": type(classifier).__name__,
                    "seed": 42,
                    "cv_folds": 3,
                    **search.best_params_,
                }
            )  # Параметры.
            mlflow.log_metric("cv_roc_auc", score)  # Результат поиска.
            mlflow.sklearn.log_model(
                search.best_estimator_,
                name="pipeline",
                pyfunc_predict_fn="predict_proba",  # MLflow возвращает вероятности.
                code_paths=["src"],  # Прикладываем собственные преобразования.
                skops_trusted_types=[
                    "numpy.dtype",
                    "src.features.Features",
                    "src.model.numeric_columns",
                    "sklearn.tree._tree.Tree",  # Деревья, обученные этим же скриптом; не внешний файл.
                ],  # Только типы нашего обученного pipeline.
                pip_requirements="requirements.txt",  # Фиксированные версии окружения.
                signature=infer_signature(X, search.best_estimator_.predict_proba(X)),
                input_example=X.head(2),
            )  # Модель каждого опыта.
            results.append(
                {
                    "name": name,
                    "cv_roc_auc": score,
                    "run_id": run.info.run_id,
                    "params": search.best_params_,
                }
            )  # Таблица сравнения.
            if score > best_score:
                best_model, best_score, best_run = (
                    search.best_estimator_,
                    score,
                    run.info.run_id,
                )  # Select using train CV only. / Выбор по train CV.
    test = pd.read_csv(
        "data/processed/test.csv"
    )  # Evaluate test after model selection. / Открываем test после выбора.
    validate(test)  # Проверяем контрольную выборку.
    probability = best_model.predict_proba(test[FEATURES])[:, 1]  # Вероятность дефолта.
    scores = metrics(test[TARGET], probability)  # Итоговые метрики.
    RocCurveDisplay.from_predictions(
        test[TARGET], probability
    )  # ROC — качество при разных порогах.
    plt.savefig("reports/roc.png", bbox_inches="tight")  # Артефакт для защиты.
    plt.close()  # Освобождаем график.
    joblib.dump(best_model, "models/pipeline.joblib")  # Готовая модель API.
    Path("reports/metrics.json").write_text(
        json.dumps(scores, indent=2)
    )  # DVC metrics.
    Path("reports/experiments.json").write_text(
        json.dumps(results, indent=2)
    )  # Пять опытов.
    with mlflow.start_run(run_id=best_run):
        mlflow.log_metrics(
            {f"test_{k}": v for k, v in scores.items()}
        )  # Test только победителя.
        mlflow.log_artifact("reports/roc.png")  # ROC в tracking.
        mlflow.log_artifact("reports/experiments.json")  # Сравнение экспериментов.
    print(json.dumps(scores, indent=2))  # Показываем результат.


if __name__ == "__main__":
    train()  # Запуск этапа обучения.
