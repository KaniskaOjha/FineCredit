"""FastAPI scoring service."""
from pathlib import Path
from typing import Literal
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from .feature_engineering import engineer_features
from .explain import reason_codes
from .train import ROOT

app = FastAPI(title="FinCredit Inclusion Scoring API", version="1.0.0")
_artifact = None


class Applicant(BaseModel):
    user_id: str = "API_APPLICANT"
    age: int = Field(ge=18, le=100)
    gender: Literal["Male", "Female", "Other"]
    region: Literal["Urban", "Rural"]
    mobile_money_txn_count_30d: float = Field(ge=0)
    mobile_money_avg_balance: float = Field(ge=0)
    mobile_money_inflow_std: float = Field(ge=0)
    mobile_money_outflow_std: float = Field(ge=0)
    mobile_money_unique_contacts: float = Field(ge=0)
    utility_on_time_payments_12m: float = Field(ge=0, le=12)
    utility_late_payments_12m: float = Field(ge=0, le=12)
    utility_avg_days_late: float = Field(ge=0)
    telco_airtime_topups_30d: float = Field(ge=0)
    telco_avg_topup_amount: float = Field(ge=0)
    telco_topup_trend: float = Field(ge=-1, le=1)
    telco_location_stability: float = Field(ge=0, le=1)
    rent_payment_regularity: float = Field(ge=0, le=1)


def artifact():
    global _artifact
    if _artifact is None:
        path = ROOT / "models" / "fair_credit_model.pkl"
        if not path.exists():
            raise HTTPException(503, "Model is not trained. Run python -m src.train first.")
        _artifact = joblib.load(path)
    return _artifact


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_ready": (ROOT / "models" / "fair_credit_model.pkl").exists()}


@app.post("/score")
def score(applicant: Applicant) -> dict:
    model = artifact()
    raw = applicant.model_dump()
    engineered = engineer_features(pd.DataFrame([raw]))
    transformed = model["preprocessor"].transform(engineered[model["feature_names"]])
    probability = float(model["probability_model"].predict_proba(transformed)[0, 1])
    fair_decision = int(model["fair_model"].predict(transformed)[0])
    decision = "approve" if fair_decision and probability >= 0.60 else ("review" if probability >= 0.40 else "reject")
    fairness = {"group": f"{applicant.gender}_{applicant.region}", "status": "monitored"}
    report_path = ROOT / "reports" / "fairness_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        fairness["gender_region_demographic_parity_difference"] = report.get("gender_region_demographic_parity_difference")
        fairness["gender_region_equalized_odds_difference"] = report.get("gender_region_equalized_odds_difference")
    return {
        "user_id": applicant.user_id, "credit_score": round(300 + 550 * probability),
        "repayment_probability": round(probability, 4), "decision": decision,
        "top_reasons": reason_codes(raw), "fairness_check": fairness,
    }
