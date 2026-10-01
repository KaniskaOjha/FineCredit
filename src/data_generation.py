"""Generate realistic synthetic alternative-credit data."""
from pathlib import Path
import numpy as np
import pandas as pd

RANDOM_SEED = 42
RAW_COLUMNS = [
    "user_id", "age", "gender", "region", "mobile_money_txn_count_30d",
    "mobile_money_avg_balance", "mobile_money_inflow_std", "mobile_money_outflow_std",
    "mobile_money_unique_contacts", "utility_on_time_payments_12m", "utility_late_payments_12m",
    "utility_avg_days_late", "telco_airtime_topups_30d", "telco_avg_topup_amount",
    "telco_topup_trend", "telco_location_stability", "rent_payment_regularity", "loan_repaid",
]


def generate_data(n_users: int = 10_000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    gender = rng.choice(["Male", "Female", "Other"], n_users, p=[0.48, 0.48, 0.04])
    region = rng.choice(["Urban", "Rural"], n_users, p=[0.58, 0.42])
    age = np.clip(rng.normal(35, 10, n_users).round(), 18, 75).astype(int)
    txn_count = rng.poisson(np.where(region == "Urban", 30, 20)).clip(0, 120)
    avg_balance = np.maximum(20, rng.lognormal(np.where(region == "Urban", 5.1, 4.6), 0.7))
    inflow_std = np.maximum(1, rng.lognormal(3.4, 0.8, n_users))
    outflow_std = np.maximum(1, rng.lognormal(3.3, 0.8, n_users))
    unique_contacts = rng.poisson(np.where(region == "Urban", 8, 5)).clip(0, 40)
    utility_on_time = rng.binomial(12, rng.beta(8, 2, n_users))
    utility_late = rng.binomial(12 - utility_on_time, rng.beta(2, 7, n_users))
    avg_days_late = np.where(utility_late > 0, rng.gamma(2.0, 4.0, n_users), 0).round(1)
    topups = rng.poisson(np.where(region == "Urban", 8, 6)).clip(0, 40)
    topup_amount = np.maximum(1, rng.lognormal(3.2, 0.55, n_users))
    topup_trend = np.clip(rng.normal(0.02, 0.18, n_users), -0.8, 0.8)
    location_stability = rng.beta(8, 2, n_users) * 0.8 + (region == "Rural") * 0.05
    rent_regularity = rng.beta(8, 2, n_users)

    risk_signal = (
        0.035 * txn_count + 0.12 * utility_on_time - 0.18 * utility_late
        - 0.025 * avg_days_late + 0.09 * topups + 1.4 * rent_regularity
        + 0.9 * location_stability + 0.35 * topup_trend
        - 0.0015 * (inflow_std + outflow_std) + 0.0008 * avg_balance
        + rng.normal(0, 1.25, n_users)
    )
    probability = 1 / (1 + np.exp(-(risk_signal - 3.4)))
    loan_repaid = rng.binomial(1, probability)
    data = pd.DataFrame({
        "user_id": [f"U{i:06d}" for i in range(1, n_users + 1)], "age": age,
        "gender": gender, "region": region, "mobile_money_txn_count_30d": txn_count,
        "mobile_money_avg_balance": avg_balance.round(2), "mobile_money_inflow_std": inflow_std.round(2),
        "mobile_money_outflow_std": outflow_std.round(2), "mobile_money_unique_contacts": unique_contacts,
        "utility_on_time_payments_12m": utility_on_time, "utility_late_payments_12m": utility_late,
        "utility_avg_days_late": avg_days_late, "telco_airtime_topups_30d": topups,
        "telco_avg_topup_amount": topup_amount.round(2), "telco_topup_trend": topup_trend.round(3),
        "telco_location_stability": location_stability.round(3), "rent_payment_regularity": rent_regularity.round(3),
        "loan_repaid": loan_repaid,
    })
    missing_cols = ["mobile_money_avg_balance", "utility_avg_days_late", "telco_topup_trend", "rent_payment_regularity"]
    for col in missing_cols:
        data.loc[rng.random(n_users) < 0.02, col] = np.nan
    return data[RAW_COLUMNS]


def main() -> None:
    output = Path(__file__).resolve().parents[1] / "data" / "credit_data.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    generate_data().to_csv(output, index=False)
    print(f"Saved {len(generate_data())} rows to {output}")


if __name__ == "__main__":
    main()
