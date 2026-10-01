"""Feature engineering for alternative credit signals."""
import numpy as np
import pandas as pd

TARGET = "loan_repaid"
ID_COLUMNS = ["user_id"]
CATEGORICAL_COLUMNS = ["gender", "region"]


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy()
    frame["transaction_regularity"] = frame["mobile_money_txn_count_30d"] / (frame["mobile_money_unique_contacts"] + 1)
    frame["balance_stability"] = frame["mobile_money_avg_balance"] / (frame["mobile_money_inflow_std"] + frame["mobile_money_outflow_std"] + 1)
    frame["payment_punctuality"] = frame["utility_on_time_payments_12m"] / (frame["utility_on_time_payments_12m"] + frame["utility_late_payments_12m"] + 1)
    frame["income_trend"] = frame["telco_topup_trend"] * np.log1p(frame["telco_avg_topup_amount"])
    frame["utility_reliability"] = frame["utility_on_time_payments_12m"] - frame["utility_late_payments_12m"]
    frame["cashflow_consistency"] = 1 / (1 + frame["mobile_money_inflow_std"] + frame["mobile_money_outflow_std"])
    frame["digital_activity"] = frame["mobile_money_txn_count_30d"] + frame["telco_airtime_topups_30d"]
    return frame


def feature_columns(data: pd.DataFrame) -> list[str]:
    return [column for column in data.columns if column not in ID_COLUMNS + [TARGET]]
