
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from train import load_and_clean_data, build_pipelines


MODEL_PATH = Path("models/heart_disease_pipeline.joblib")


def test_model_loads():
    """Check that the saved model can be loaded."""
    assert MODEL_PATH.exists(), f"Model not found: {MODEL_PATH}"

    model = joblib.load(MODEL_PATH)
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


def test_predict_proba_returns_valid_values():
    """Check that predicted probabilities are between zero and one."""
    df = load_and_clean_data()
    X = df.drop(columns=["target"])
    model = joblib.load(MODEL_PATH)

    probabilities = model.predict_proba(X.iloc[:10])

    assert probabilities.shape[0] == 10
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_training_smoke_test():
    """Check that a pipeline can train and predict on a small sample."""
    df = load_and_clean_data()
    X = df.drop(columns=["target"])
    y = df["target"]

    X_sample, _, y_sample, _ = train_test_split(
        X,
        y,
        train_size=60,
        random_state=42,
        stratify=y,
    )

    logistic_model, _ = build_pipelines()
    logistic_model.fit(X_sample, y_sample)
    predictions = logistic_model.predict(X_sample)

    assert len(predictions) == len(y_sample)
    assert set(predictions).issubset({0, 1})
