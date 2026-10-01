# Kaniska FineCredit

Kaniska FineCredit is an end-to-end reference system for estimating micro-loan repayment likelihood from consented alternative data: mobile-money activity, utility payments, telco behavior, and rent regularity. It includes Fairlearn Demographic Parity constraints, group-level audits, explainability, a FastAPI service, and a Streamlit dashboard.

The project submission includes [`Kaniska_FineCredit.ipynb`](Kaniska_FineCredit.ipynb) and [`Kaniska_FineCredit_ProjectReport.docx`](Kaniska_FineCredit_ProjectReport.docx). The bundled dataset is the reproducible synthetic file [`data/credit_data.csv`](data/credit_data.csv); no real applicant data is used.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.data_generation
python -m src.train
python -m src.evaluate
python -m src.fairness_audit
python -m src.explain
uvicorn src.api:app --reload
# separate terminal
streamlit run app/app.py
```

API docs are available at `http://127.0.0.1:8000/docs`; the dashboard runs at `http://localhost:8501`.

## Pipeline

1. `src/data_generation.py` creates 10,000 reproducible synthetic applicants at `data/credit_data.csv` with realistic relationships, noise, and missing values.
2. `src/feature_engineering.py` creates transaction regularity, balance stability, payment punctuality, income trend, cash-flow consistency, and digital activity.
3. `src/train.py` compares Logistic Regression, XGBoost, and LightGBM with SMOTE, then trains a Fairlearn ExponentiatedGradient model constrained by Demographic Parity over intersectional gender-region groups. The artifact is `models/fair_credit_model.pkl`.
4. `src/evaluate.py` writes standard metrics and `reports/roc_curve.png`.
5. `src/fairness_audit.py` writes approval rates and disparity metrics to `reports/fairness_report.json`.
6. `src/explain.py` writes a global SHAP plot to `reports/shap_summary.png` and provides reason codes for API scoring.

## API example

`POST /score` accepts the applicant fields listed in `src/api.py` and returns a 300-850 score, repayment probability, approve/review/reject decision, reasons, and monitored fairness metrics.

## Ethics and limitations

This is a synthetic reference implementation, not a production lending decision system. Use only data collected with informed consent and appropriate legal basis; minimize retention, secure sensitive attributes, document data provenance, monitor drift, and provide notice, appeal, and human review. Fairness metrics are diagnostic rather than proof of legal compliance. The model is intended to expand responsible access, not to exclude people without traditional credit histories.
