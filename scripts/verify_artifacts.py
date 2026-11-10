"""Verify bundled model and tracking. / Проверить модель и журнал экспериментов."""

import hashlib
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from src.schema import FEATURES, TARGET
from src.train import metrics


def main():
    train = pd.read_csv("data/processed/train.csv")
    test = pd.read_csv("data/processed/test.csv")
    assert len(train) == 24000 and len(test) == 6000
    assert not set(train.ID) & set(test.ID)
    metadata = json.loads(Path("data/raw/source.json").read_text())
    assert (
        hashlib.sha256(Path("data/raw/source.zip").read_bytes()).hexdigest()
        == metadata["sha256"]
    )
    model = joblib.load("models/pipeline.joblib")
    actual = metrics(test[TARGET], model.predict_proba(test[FEATURES])[:, 1])
    expected = json.loads(Path("reports/metrics.json").read_text())
    assert all(np.isclose(actual[k], expected[k], atol=1e-12) for k in actual)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()
    experiments = json.loads(Path("reports/experiments.json").read_text())
    assert len(experiments) >= 5
    for experiment in experiments:
        run = client.get_run(experiment["run_id"])
        assert run.info.status == "FINISHED"
        assert np.isclose(run.data.metrics["cv_roc_auc"], experiment["cv_roc_auc"])
    best = max(experiments, key=lambda x: x["cv_roc_auc"])
    # Find the logged model without relying on an absolute location. / Находим модель без абсолютного пути.
    logged = client.search_logged_models(
        experiment_ids=[client.get_run(best["run_id"]).info.experiment_id]
    )
    selected = next(m for m in logged if m.source_run_id == best["run_id"])
    tracked = mlflow.sklearn.load_model(f"models:/{selected.model_id}")
    np.testing.assert_allclose(
        tracked.predict_proba(test[FEATURES].head(10)),
        model.predict_proba(test[FEATURES].head(10)),
    )
    report = {
        "rows_train": len(train),
        "rows_test": len(test),
        "disjoint_ids": True,
        "source_sha256_matches": True,
        "finished_runs": len(experiments),
        "mlflow_model_matches_joblib": True,
        "metrics": actual,
    }
    Path("reports/artifact-verification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
