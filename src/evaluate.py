"""Evaluate the saved model and write performance/fairness artifacts."""
from pathlib import Path
import json
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, RocCurveDisplay
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from .feature_engineering import engineer_features
from .train import ROOT, load_data


def evaluate() -> dict:
    artifact = joblib.load(ROOT / "models" / "fair_credit_model.pkl")
    data = engineer_features(load_data())
    X = artifact["preprocessor"].transform(data[artifact["feature_names"]])
    sensitive = data["gender"].fillna("Unknown") + "_" + data["region"].fillna("Unknown")
    y_true = data["loan_repaid"].astype(int)
    y_pred = artifact["fair_model"].predict(X)
    probabilities = artifact["probability_model"].predict_proba(X)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)), "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)), "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)), "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "demographic_parity_difference": float(demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive)),
        "equalized_odds_difference": float(equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive)),
        "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    with open(reports / "evaluation.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    RocCurveDisplay.from_predictions(y_true, probabilities)
    plt.title("Fair Credit Model ROC Curve")
    plt.tight_layout()
    plt.savefig(reports / "roc_curve.png", dpi=140)
    plt.close()
    print(json.dumps({key: value for key, value in metrics.items() if key != "classification_report"}, indent=2))
    return metrics


if __name__ == "__main__":
    evaluate()
