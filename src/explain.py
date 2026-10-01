"""SHAP explanations and human-readable reason codes."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .feature_engineering import engineer_features
from .train import ROOT, load_data

REASON_LABELS = {
    "transaction_regularity": "consistent mobile-money activity", "balance_stability": "stable account balance",
    "payment_punctuality": "on-time utility payments", "income_trend": "positive top-up trend",
    "utility_reliability": "utility payment reliability", "cashflow_consistency": "cash-flow consistency",
    "digital_activity": "regular digital activity", "utility_late_payments_12m": "late utility payments",
    "utility_avg_days_late": "days late on utility payments", "mobile_money_inflow_std": "variable incoming cash flow",
    "mobile_money_outflow_std": "variable outgoing cash flow", "rent_payment_regularity": "rent payment regularity",
}


def shap_summary(output_path: str | Path = ROOT / "reports" / "shap_summary.png") -> Path:
    import shap
    import matplotlib.pyplot as plt
    artifact = joblib.load(ROOT / "models" / "fair_credit_model.pkl")
    data = engineer_features(load_data())
    sample = data[artifact["feature_names"]].sample(min(1000, len(data)), random_state=42)
    transformed = artifact["preprocessor"].transform(sample)
    explainer = shap.Explainer(artifact["probability_model"], transformed)
    values = explainer(transformed)
    plt.figure()
    shap.plots.bar(values, max_display=15, show=False)
    Path(output_path).parent.mkdir(exist_ok=True)
    plt.tight_layout(); plt.savefig(output_path, dpi=140, bbox_inches="tight"); plt.close()
    return Path(output_path)


def reason_codes(applicant: dict, top_n: int = 3) -> list[str]:
    artifact = joblib.load(ROOT / "models" / "fair_credit_model.pkl")
    row = engineer_features(pd.DataFrame([applicant]))
    values = row[artifact["feature_names"]].iloc[0]
    reasons = []
    for feature, value in values.items():
        if pd.notna(value) and feature in REASON_LABELS:
            reasons.append((float(value), REASON_LABELS[feature]))
    reasons.sort(key=lambda item: abs(item[0]), reverse=True)
    return [label for _, label in reasons[:top_n]]
