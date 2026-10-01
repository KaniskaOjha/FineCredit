"""Streamlit dashboard for FinCredit."""
from pathlib import Path
import json
import sys
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering import engineer_features

ROOT = PROJECT_ROOT

st.set_page_config(page_title="FinCredit", page_icon="F", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv(ROOT / "data" / "credit_data.csv")

@st.cache_data
def load_json(name):
    path = ROOT / "reports" / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

st.title("FinCredit")
st.caption("Alternative-data credit access with transparent, fairness-aware decisions")
data = load_data()
metrics = load_json("evaluation.json")
audit = load_json("fairness_report.json")
page = st.sidebar.radio("Workspace", ["Overview", "Applicant Scoring", "Fairness Dashboard", "Model Explanation"])

if page == "Overview":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Applicants", f"{len(data):,}")
    c2.metric("Observed repayment", f"{data.loan_repaid.mean():.1%}")
    c3.metric("Model F1", f"{metrics.get('f1', 0):.3f}")
    c4.metric("Parity difference", f"{metrics.get('gender_region_demographic_parity_difference', 0):.3f}")
    st.subheader("Portfolio snapshot")
    st.plotly_chart(px.histogram(data, x="age", color="region", nbins=25, barmode="overlay"), use_container_width=True)
    st.info("This tool supports responsible financial inclusion. Use consented data, protect privacy, and provide human review for adverse decisions.")
elif page == "Applicant Scoring":
    st.subheader("Applicant Scoring")
    st.info(
        "Decision policy: **Approve** when Fairlearn approves and repayment probability is at least 60%; "
        "**Review** when probability is 40%-59.99%; **Reject** when probability is below 40%."
    )
    with st.form("score_form"):
        cols = st.columns(3)
        age = cols[0].number_input("Age", 18, 100, 35)
        gender = cols[1].selectbox("Gender", ["Female", "Male", "Other"])
        region = cols[2].selectbox("Region", ["Urban", "Rural"])
        txn = cols[0].number_input("Mobile-money transactions (30d)", 0, 200, 24)
        balance = cols[1].number_input("Average balance", 0.0, 100000.0, 250.0)
        inflow = cols[2].number_input("Inflow variability", 0.0, 10000.0, 30.0)
        outflow = cols[0].number_input("Outflow variability", 0.0, 10000.0, 30.0)
        contacts = cols[1].number_input("Unique contacts", 0, 100, 8)
        on_time = cols[2].number_input("Utility payments on time", 0, 12, 9)
        late = cols[0].number_input("Utility payments late", 0, 12, 2)
        days_late = cols[1].number_input("Average days late", 0.0, 100.0, 2.0)
        topups = cols[2].number_input("Airtime top-ups (30d)", 0, 100, 8)
        topup_amount = cols[0].number_input("Average top-up amount", 0.0, 10000.0, 25.0)
        trend = cols[1].number_input("Top-up trend", -1.0, 1.0, 0.05)
        stability = cols[2].slider("Location stability", 0.0, 1.0, 0.8)
        rent = cols[0].slider("Rent regularity", 0.0, 1.0, 0.8)
        submitted = st.form_submit_button("Score applicant")
    if submitted:
        from src.api import score, Applicant
        result = score(Applicant(age=age, gender=gender, region=region, mobile_money_txn_count_30d=txn, mobile_money_avg_balance=balance, mobile_money_inflow_std=inflow, mobile_money_outflow_std=outflow, mobile_money_unique_contacts=contacts, utility_on_time_payments_12m=on_time, utility_late_payments_12m=late, utility_avg_days_late=days_late, telco_airtime_topups_30d=topups, telco_avg_topup_amount=topup_amount, telco_topup_trend=trend, telco_location_stability=stability, rent_payment_regularity=rent))
        score_col, probability_col = st.columns(2)
        score_col.metric("Credit score", result["credit_score"])
        probability_col.metric("Repayment probability", f"{result['repayment_probability']:.1%}")
        if result["decision"] == "approve":
            st.success(result["decision"].upper())
        else:
            st.warning(result["decision"].upper())
        st.write("Reasons", ", ".join(result["top_reasons"]))
elif page == "Fairness Dashboard":
    st.subheader("Fairness Dashboard")
    rates = audit.get("approval_rates", {})
    for dimension in ["gender", "region", "gender_region"]:
        values = pd.DataFrame(list(rates.get(dimension, {}).items()), columns=["group", "approval_rate"])
        st.write(dimension.replace("_", " ").title())
        st.plotly_chart(px.bar(values, x="group", y="approval_rate", range_y=[0, 1]), use_container_width=True)
    st.json({key: value for key, value in audit.items() if "difference" in key})
else:
    st.subheader("Model Explanation")
    image = ROOT / "reports" / "shap_summary.png"
    if image.exists(): st.image(str(image), caption="Global feature importance")
    else: st.warning("Run python -m src.explain to generate the SHAP summary.")
