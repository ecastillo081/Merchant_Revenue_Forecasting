import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from functions.forecast_methods import (
    H,
    naive_forecast,
    seasonal_naive_forecast,
    sma_forecast,
    wma_forecast,
)

RAW_PATH = ROOT / "data" / "raw" / "merchant_monthly_revenue.xlsx"
TRANSFORMED_DIR = ROOT / "data" / "transformed"

# Illustrative single-merchant example (not the full nine-model portfolio benchmark)
merchant_revenue_df = pd.read_excel(RAW_PATH)

df = merchant_revenue_df.copy()
df["date"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp()

m_id = "M001"

d = df[df["merchant_id"] == m_id].set_index("date")
y = d["revenue"].astype(float)
y_train, y_test = y.iloc[:-H], y.iloc[-H:]

ses = SimpleExpSmoothing(y_train).fit(optimized=True)
ses_fc = ses.forecast(H)

holt = ExponentialSmoothing(y_train, trend="add", seasonal=None).fit(optimized=True)
holt_fc = holt.forecast(H)

hw = ExponentialSmoothing(
    y_train, trend="add", seasonal="mul", seasonal_periods=12
).fit(optimized=True)
hw_fc = hw.forecast(H)

sarima = SARIMAX(
    y_train,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
    enforce_stationarity=False,
    enforce_invertibility=False,
).fit(disp=False)
sarima_fc = sarima.forecast(H)

preds = {
    "Naive": naive_forecast(y_train, H),
    "Seasonal Naive": seasonal_naive_forecast(y_train, H, 12),
    "SMA(3)": sma_forecast(y_train, H, 3),
    "WMA": wma_forecast(y_train, H, (1, 2, 3)),
    "SES": ses_fc,
    "Holt": holt_fc,
    "Holt-Winters": hw_fc,
    "SARIMA": sarima_fc,
}


def forecasts_to_df(preds_dict, y_test_index):
    df_out = pd.DataFrame(index=y_test_index)
    for name, fc in preds_dict.items():
        if fc is None:
            continue
        fc = np.asarray(fc, dtype=float).ravel()
        if len(fc) != len(y_test_index):
            raise ValueError(f"{name}: expected {len(y_test_index)} steps, got {len(fc)}")
        df_out[name] = fc
    return df_out


forecasts_df = forecasts_to_df(preds, y_test.index)
forecasts_df.insert(0, "Actual", y_test.values)
forecasts_df.insert(0, "merchant_id", m_id)

TRANSFORMED_DIR.mkdir(parents=True, exist_ok=True)
forecasts_df.to_excel(TRANSFORMED_DIR / f"{m_id}_forecast.xlsx")
