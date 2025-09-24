# Synthetic dataset generator for merchant monthly revenue/orders/marketing spend
import numpy as np
import pandas as pd
from datetime import datetime
import math
import json

np.random.seed(42)

# -----------------------
# Config
# -----------------------
N_MERCHANTS = 50
START = "2020-01-01"
END = "2024-12-01"  # inclusive month start
freq = "MS"  # Month Start
dates = pd.date_range(start=START, end=END, freq=freq)
n_months = len(dates)

# Merchant metadata
regions = ["West", "Midwest", "Northeast", "South"]
verticals = ["Retail", "Food & Bev", "Services", "Digital Goods"]
merchant_ids = [f"M{str(i + 1).zfill(3)}" for i in range(N_MERCHANTS)]
meta = []
for m in merchant_ids:
    region = np.random.choice(regions, p=[0.35, 0.2, 0.25, 0.2])
    vertical = np.random.choice(verticals, p=[0.4, 0.25, 0.25, 0.1])
    # base monthly revenue level ($)
    base_rev = np.random.uniform(20000, 150000)
    # average order value (~$20-$120 depending on vertical)
    aov_base = {
        "Retail": np.random.uniform(40, 120),
        "Food & Bev": np.random.uniform(15, 45),
        "Services": np.random.uniform(60, 200),
        "Digital Goods": np.random.uniform(10, 60),
    }[vertical]
    # trend type: growth, flat, mild decline
    trend_type = np.random.choice(["growth", "flat", "decline"], p=[0.55, 0.3, 0.15])
    trend_pct = {
        "growth": np.random.uniform(0.03, 0.20),  # annual growth 3%–20%
        "flat": np.random.uniform(-0.01, 0.01),  # -1%–1%
        "decline": np.random.uniform(-0.15, -0.03)  # -15%–-3%
    }[trend_type]
    # marketing aggressiveness (share of revenue invested in marketing)
    mkt_ratio = np.clip(np.random.normal(0.08, 0.03), 0.02, 0.2)  # 2%–20%
    # elasticity (diminishing returns via log(1+spend))
    elasticity = np.random.uniform(50, 200)  # revenue dollars per log-dollar spend
    # seasonality strength
    seas_strength = np.random.uniform(0.05, 0.25)  # 5%–25% peak-to-trough
    # promotion propensity
    promo_prob = np.random.uniform(0.06, 0.18)  # chance of a promo month
    meta.append({
        "merchant_id": m,
        "region": region,
        "vertical": vertical,
        "base_rev": base_rev,
        "aov_base": aov_base,
        "trend_type": trend_type,
        "trend_pct_annual": trend_pct,
        "mkt_ratio": mkt_ratio,
        "elasticity": elasticity,
        "seas_strength": seas_strength,
        "promo_prob": promo_prob
    })
meta_df = pd.DataFrame(meta)

# Global monthly seasonality pattern (e.g., Q4 spike, summer dip) normalized to mean 1.0
month = np.array([d.month for d in dates])
# Base seasonal multipliers: Nov/Dec up, Jan down a bit
base_season = {
    1: 0.93, 2: 0.95, 3: 1.00, 4: 1.02, 5: 1.05, 6: 0.98,
    7: 0.96, 8: 1.00, 9: 1.03, 10: 1.06, 11: 1.15, 12: 1.20
}
season_vec = np.array([base_season[m] for m in month])
season_vec = season_vec / season_vec.mean()  # normalize mean to 1.0

# Macro factor (optional): mild shock & recovery (e.g., pandemic-like dip)
macro = np.ones(n_months)
# Dip around 2020-04 to 2020-06, then rebound
for i, d in enumerate(dates):
    if datetime(2020, 3, 1) <= d <= datetime(2020, 6, 1):
        macro[i] = 0.92 + 0.02 * (i % 3)  # rough dip
    elif datetime(2020, 7, 1) <= d <= datetime(2020, 12, 1):
        macro[i] = 1.02 + 0.01 * ((i - 6) % 6)  # recovery
macro = macro / macro.mean()

rows = []
for _, r in meta_df.iterrows():
    # merchant-specific seasonality intensity
    seas = 1 + (season_vec - 1) * r["seas_strength"]
    # additive monthly trend factor (compound annual to monthly)
    monthly_trend = (1 + r["trend_pct_annual"]) ** (np.arange(n_months) / 12.0)
    # random walk noise to mimic operational variability
    noise = np.exp(np.random.normal(0, 0.08, n_months))  # lognormal ~8% sigma
    # baseline demand before marketing & promos
    baseline = r["base_rev"] * monthly_trend * seas * macro

    # marketing spend as a share of revenue with some random schedule and ramp
    # Start with base plan ~ mkt_ratio * previous revenue (recursive); initialize
    mkt_spend = np.zeros(n_months)
    rev = np.zeros(n_months)
    # promos: some months have extra spend & extra lift
    promos = (np.random.rand(n_months) < r["promo_prob"]).astype(int)
    promo_lift = 1 + promos * np.random.uniform(0.05, 0.30, n_months)  # +5%–30% lift

    for t in range(n_months):
        # plan spend ~ target ratio of last month's revenue, with some noise
        if t == 0:
            planned_spend = r["mkt_ratio"] * r["base_rev"] * np.random.uniform(0.7, 1.3)
        else:
            planned_spend = r["mkt_ratio"] * max(rev[t - 1], 1.0) * np.random.uniform(0.8, 1.25)
        # add promo budget if promo month
        if promos[t] == 1:
            planned_spend *= np.random.uniform(1.2, 1.8)
        # ensure reasonable bounds
        planned_spend = np.clip(planned_spend, 1000, 0.35 * baseline[t])
        mkt_spend[t] = planned_spend

        # revenue as: baseline * promo_lift + elasticity * log(1 + mkt_spend) + noise
        rev[t] = baseline[t] * promo_lift[t] + r["elasticity"] * math.log1p(mkt_spend[t])
        # apply idiosyncratic noise (multiplicative)
        rev[t] *= noise[t]
        # floor at small positive
        rev[t] = max(rev[t], 5000.0)

    # orders from revenue and AOV with small drift
    aov_drift = np.cumprod(1 + np.random.normal(0.001, 0.003, n_months))  # mild drift
    aov = r["aov_base"] * aov_drift
    orders = np.maximum(np.round(rev / np.maximum(aov, 5.0) + np.random.normal(0, 3, n_months)), 1).astype(int)

    # compute simple derived KPIs
    cac = mkt_spend / np.maximum(orders, 1)  # $ per order
    mktg_rev_ratio = mkt_spend / np.maximum(rev, 1)

    for t, d in enumerate(dates):
        rows.append({
            "date": d.strftime("%Y-%m-%d"),
            "merchant_id": r["merchant_id"],
            "region": r["region"],
            "vertical": r["vertical"],
            "revenue": round(rev[t], 2),
            "orders": int(orders[t]),
            "avg_order_value": round(aov[t], 2),
            "marketing_spend": round(float(mkt_spend[t]), 2),
            "promo_month": int(promos[t]),
            "seasonal_index": round(float(seas[t]), 4),
            "macro_index": round(float(macro[t]), 4),
            "mktg_rev_ratio": round(float(mktg_rev_ratio[t]), 4),
            "cac_per_order": round(float(cac[t]), 2)
        })

data = pd.DataFrame(rows)

data_dict = {
    "date": "Month start date (YYYY-MM-DD)",
    "merchant_id": "Unique merchant identifier (M001..M050)",
    "region": "US region (West, Midwest, Northeast, South)",
    "vertical": "Industry vertical (Retail, Food & Bev, Services, Digital Goods)",
    "revenue": "Monthly gross sales revenue in USD",
    "orders": "Monthly order count (integer)",
    "avg_order_value": "Average order value in USD",
    "marketing_spend": "Monthly marketing spend in USD",
    "promo_month": "1 if a promo ran this month, else 0",
    "seasonal_index": "Merchant-adjusted seasonal multiplier (mean ~1)",
    "macro_index": "Global macro factor multiplier (mean ~1)",
    "mktg_rev_ratio": "Actual marketing spend / revenue",
    "cac_per_order": "Marketing spend / orders (CAC)"
}

# export
xlsx_path = "../raw/merchant_monthly_revenue.xlsx"
data_dict_path = "../raw/merchant_monthly_data_dictionary.json"

data.to_excel(xlsx_path, index=False)

with open(data_dict_path, "w") as f:
    json.dump(data_dict, f, indent=2)
