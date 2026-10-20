
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from train import load_and_clean_data, build_pipelines

# Output folder
Path("results").mkdir(exist_ok=True)

# Load the same cleaned dataset used by the training pipeline
df = load_and_clean_data()
X = df.drop(columns=["target"])
y = df["target"]

# Use reproducible 5-fold stratified cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Metrics required for the assignment
scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}

logistic_model, random_forest_model = build_pipelines()

models = {
    "Logistic Regression": logistic_model,
    "Random Forest": random_forest_model,
}

rows = []

for model_name, model in models.items():
    print(f"\nEvaluating {model_name}...")

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        error_score="raise",
    )

    row = {"Model": model_name}

    for metric in scoring:
        values = scores[f"test_{metric}"]

        row[f"{metric.capitalize()} Mean"] = values.mean()
        row[f"{metric.capitalize()} Std"] = values.std(ddof=1)

    rows.append(row)

# Save the complete numerical results
results = pd.DataFrame(rows)
output_path = Path("results/cross_validation_summary.csv")
results.to_csv(output_path, index=False)

# Print a report-friendly table
print("\n5-Fold Cross-Validation Results (Mean ± Standard Deviation)")
print("=" * 75)

for _, row in results.iterrows():
    print(f"\n{row['Model']}")
    for metric in scoring:
        name = metric.capitalize()
        print(
            f"{name:10s}: "
            f"{row[f'{name} Mean']:.3f} ± "
            f"{row[f'{name} Std']:.3f}"
        )

print(f"\nCSV saved to: {output_path}")
