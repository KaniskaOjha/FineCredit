"""Train benchmark models and a Fairlearn-constrained production model."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from fairlearn.reductions import DemographicParity, ExponentiatedGradient
from .feature_engineering import engineer_features, feature_columns
from .preprocessing import build_preprocessor

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

ROOT = Path(__file__).resolve().parents[1]


def load_data() -> pd.DataFrame:
    path = ROOT / "data" / "credit_data.csv"
    if not path.exists():
        from .data_generation import generate_data
        path.parent.mkdir(exist_ok=True)
        generate_data().to_csv(path, index=False)
    return pd.read_csv(path)


def train() -> dict:
    data = engineer_features(load_data())
    features = feature_columns(data)
    X = data[features]
    y = data["loan_repaid"].astype(int)
    sensitive = data["gender"].fillna("Unknown") + "_" + data["region"].fillna("Unknown")
    X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = train_test_split(
        X, y, sensitive, test_size=0.2, random_state=42, stratify=y
    )
    preprocessor = build_preprocessor(features)
    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.transform(X_test)
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train_encoded, y_train)
    benchmarks = {"logistic_regression": LogisticRegression(max_iter=1500, class_weight="balanced")}
    if XGBClassifier:
        benchmarks["xgboost"] = XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.05, subsample=0.85, colsample_bytree=0.85, eval_metric="logloss", random_state=42)
    if LGBMClassifier:
        benchmarks["lightgbm"] = LGBMClassifier(n_estimators=250, num_leaves=24, learning_rate=0.05, class_weight="balanced", random_state=42, verbosity=-1)
    benchmark_scores = {}
    for name, model in benchmarks.items():
        model.fit(X_resampled, y_resampled)
        benchmark_scores[name] = float(roc_auc_score(y_test, model.predict_proba(X_test_encoded)[:, 1]))
    fair_base = LogisticRegression(max_iter=1500, class_weight="balanced")
    fair_model = ExponentiatedGradient(estimator=fair_base, constraints=DemographicParity(), eps=0.02)
    fair_model.fit(X_train_encoded, y_train, sensitive_features=sensitive_train)
    fair_predictions = fair_model.predict(X_test_encoded)
    fair_auc = float(roc_auc_score(y_test, fair_predictions))
    artifact = {
        "preprocessor": preprocessor, "fair_model": fair_model, "probability_model": fair_base,
        "feature_names": features, "threshold": 0.5, "benchmark_scores": benchmark_scores,
        "fair_model_auc_proxy": fair_auc, "sensitive_groups": sorted(sensitive.unique().tolist()),
    }
    # Keep the base estimator fitted on encoded data for calibrated credit-score probabilities.
    fair_base.fit(X_train_encoded, y_train)
    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(artifact, models_dir / "fair_credit_model.pkl")
    with open(models_dir / "training_metrics.json", "w", encoding="utf-8") as handle:
        json.dump({"benchmark_roc_auc": benchmark_scores, "fair_decision_auc_proxy": fair_auc}, handle, indent=2)
    print(json.dumps({"saved": str(models_dir / "fair_credit_model.pkl"), "benchmark_roc_auc": benchmark_scores}, indent=2))
    return artifact


if __name__ == "__main__":
    train()
