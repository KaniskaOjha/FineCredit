"""Produce group-level approval rates and disparity metrics."""
from pathlib import Path
import json
import joblib
import pandas as pd
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from .feature_engineering import engineer_features
from .train import ROOT, load_data


def audit() -> dict:
    artifact = joblib.load(ROOT / "models" / "fair_credit_model.pkl")
    data = engineer_features(load_data())
    transformed = artifact["preprocessor"].transform(data[artifact["feature_names"]])
    data["prediction"] = artifact["fair_model"].predict(transformed)
    approval_rates = {
        "gender": data.groupby("gender", dropna=False)["prediction"].mean().round(4).to_dict(),
        "region": data.groupby("region", dropna=False)["prediction"].mean().round(4).to_dict(),
        "gender_region": (data.assign(group=data["gender"] + "_" + data["region"]).groupby("group")["prediction"].mean().round(4).to_dict()),
    }
    report = {"approval_rates": approval_rates}
    for column in ["gender", "region", "gender_region"]:
        groups = data[column] if column in data else data["gender"] + "_" + data["region"]
        report[f"{column}_demographic_parity_difference"] = float(demographic_parity_difference(data["loan_repaid"], data["prediction"], sensitive_features=groups))
        report[f"{column}_equalized_odds_difference"] = float(equalized_odds_difference(data["loan_repaid"], data["prediction"], sensitive_features=groups))
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    with open(reports / "fairness_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    audit()
