"""Reproducible training pipeline for the MLOps heart disease project.

This script follows the training and model-selection logic in
notebooks/01_data_acquisition_eda.ipynb.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)
import matplotlib.pyplot as plt
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/heart-disease/"
    "processed.cleveland.data"
)
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = [
    "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"
]


def load_and_clean_data() -> pd.DataFrame:
    """Download the same UCI Cleveland dataset and apply notebook cleaning."""
    Path("data").mkdir(exist_ok=True)
    raw_path = Path("data/heart_disease_raw.csv")
    clean_path = Path("data/heart_disease_clean.csv")

    df = pd.read_csv(
        DATA_URL,
        header=None,
        names=COLUMNS,
        na_values="?",
    )
    df.to_csv(raw_path, index=False)

    df_clean = df.dropna().copy()
    # Original target: 0 = no disease, 1-4 = disease present.
    df_clean["target"] = (df_clean["target"] > 0).astype(int)

    for col in CATEGORICAL_FEATURES:
        df_clean[col] = df_clean[col].astype(int)

    df_clean.to_csv(clean_path, index=False)
    print(f"Cleaned dataset shape: {df_clean.shape}")
    print("Target distribution:")
    print(df_clean["target"].value_counts().sort_index())
    return df_clean


def build_pipelines() -> tuple[Pipeline, Pipeline]:
    """Build the preprocessing and classifier pipelines from the notebook."""
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])

    logistic_model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=2000)),
    ])
    random_forest_model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight=None,
        )),
    ])
    return logistic_model, random_forest_model


def main() -> None:
    Path("models").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    Path("mlflow_data").mkdir(exist_ok=True)

    df_clean = load_and_clean_data()
    X = df_clean.drop(columns=["target"])
    y = df_clean["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    logistic_model, random_forest_model = build_pipelines()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Baseline cross-validation metrics, matching the notebook's comparison.
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "roc_auc": "roc_auc",
    }
    baseline_rows = []
    for name, model in [
        ("Logistic Regression", logistic_model),
        ("Random Forest", random_forest_model),
    ]:
        print(f"\nBaseline cross-validation: {name}")
        scores = cross_validate(
            model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1
        )
        baseline_rows.append({
            "Model": name,
            "CV Accuracy": scores["test_accuracy"].mean(),
            "CV Precision": scores["test_precision"].mean(),
            "CV Recall": scores["test_recall"].mean(),
            "CV ROC-AUC": scores["test_roc_auc"].mean(),
        })
    pd.DataFrame(baseline_rows).to_csv(
        "results/baseline_cv_comparison.csv", index=False
    )

    # GridSearchCV configurations copied from the notebook.
    lr_grid = {
        "classifier__C": [0.01, 0.1, 1, 10],
        "classifier__class_weight": [None, "balanced"],
    }
    rf_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [None, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
    }

    searches = {
        "Logistic Regression": GridSearchCV(
            estimator=logistic_model,
            param_grid=lr_grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=True,
        ),
        "Random Forest": GridSearchCV(
            estimator=random_forest_model,
            param_grid=rf_grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=True,
        ),
    }

    tuned_rows = []
    for name, search in searches.items():
        print(f"\nTuning {name}...")
        search.fit(X_train, y_train)
        print("Best parameters:", search.best_params_)
        print("Best mean CV ROC-AUC:", round(search.best_score_, 4))

        model = search.best_estimator_
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]


                # Generate evaluation plots for this tuned model
        Path("plots").mkdir(exist_ok=True)

        # Confusion matrix
        cm_path = f"plots/confusion_matrix_{name.lower().replace(' ', '_')}.png"
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            display_labels=["No Disease", "Disease Present"],
            cmap="Blues",
        )
        plt.title(f"Confusion Matrix - {name}")
        plt.tight_layout()
        plt.savefig(cm_path, dpi=150, bbox_inches="tight")
        plt.close()

        # ROC curve
        roc_path = f"plots/roc_curve_{name.lower().replace(' ', '_')}.png"
        RocCurveDisplay.from_predictions(y_test, y_prob)
        plt.title(f"ROC Curve - {name}")
        plt.tight_layout()
        plt.savefig(roc_path, dpi=150, bbox_inches="tight")
        plt.close()
        tuned_rows.append({
            "Model": name,
            "CV ROC-AUC": float(search.best_score_),
            "Test Accuracy": accuracy_score(y_test, y_pred),
            "Test Precision": precision_score(
                y_test, y_pred, zero_division=0
            ),
            "Test Recall": recall_score(y_test, y_pred, zero_division=0),
            "Test ROC-AUC": roc_auc_score(y_test, y_prob),
            "Best Parameters": json.dumps(search.best_params_, default=str),
        })
        print(classification_report(
            y_test,
            y_pred,
            target_names=["No Disease", "Disease Present"],
            zero_division=0,
        ))

    tuned_results_df = pd.DataFrame(tuned_rows)
    tuned_results_df.to_csv(
        "results/tuned_model_comparison.csv", index=False
    )

    # Select using cross-validation ROC-AUC, not test-set performance.
    lr_search = searches["Logistic Regression"]
    rf_search = searches["Random Forest"]
    if lr_search.best_score_ >= rf_search.best_score_:
        final_model = lr_search.best_estimator_
        selected_model_name = "Logistic Regression"
        selection_score = lr_search.best_score_
    else:
        final_model = rf_search.best_estimator_
        selected_model_name = "Random Forest"
        selection_score = rf_search.best_score_

    model_path = "models/heart_disease_pipeline.joblib"
    joblib.dump(final_model, model_path)

    metadata = {
        "project": "Heart Disease Classification",
        "selected_model": selected_model_name,
        "target": {"0": "No disease", "1": "Disease present"},
        "input_features": list(X.columns),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "selection_metric": "Mean cross-validation ROC-AUC",
        "selection_score": float(selection_score),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open("models/model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    # Track each tuned candidate and its test metrics in MLflow.
    mlflow.set_tracking_uri("sqlite:///mlflow_data/mlflow.db")
    mlflow.set_experiment("Heart_Disease_Classification")
    for name, search in searches.items():
        row = next(r for r in tuned_rows if r["Model"] == name)
        with mlflow.start_run(run_name=name):
            mlflow.log_params({
                key: str(value) for key, value in search.best_params_.items()
            })
            mlflow.log_metrics({
                "cv_roc_auc": float(search.best_score_),
                "test_accuracy": float(row["Test Accuracy"]),
                "test_precision": float(row["Test Precision"]),
                "test_recall": float(row["Test Recall"]),
                "test_roc_auc": float(row["Test ROC-AUC"]),
            })
            mlflow.sklearn.log_model(
            sk_model=search.best_estimator_,
            name="model",
            skops_trusted_types=[
                "numpy.dtype",
                "sklearn.compose._column_transformer._RemainderColsList",
                "sklearn.tree._tree.Tree",
            ],
        )
            for artifact in [
                "results/baseline_cv_comparison.csv",
                "results/tuned_model_comparison.csv",
                model_path,
                "models/model_metadata.json",
                "plots/confusion_matrix_logistic_regression.png",
                "plots/roc_curve_logistic_regression.png",
                "plots/confusion_matrix_random_forest.png",
                "plots/roc_curve_random_forest.png",
            ]:
                if Path(artifact).exists():
                    mlflow.log_artifact(artifact)
                if Path(artifact).exists():
                    mlflow.log_artifact(artifact)

    print(f"\nSelected model: {selected_model_name}")
    print(f"Saved model: {model_path}")
    print("Saved metadata: models/model_metadata.json")
    print("Saved metrics: results/baseline_cv_comparison.csv")
    print("Saved metrics: results/tuned_model_comparison.csv")
    print("MLflow tracking database: mlflow_data/mlflow.db")


if __name__ == "__main__":
    main()
