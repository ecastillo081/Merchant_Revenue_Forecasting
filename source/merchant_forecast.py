import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from functions.forecast_methods import H, naive_forecast, seasonal_naive_forecast, sma_forecast, wma_forecast

# load data from data/raw/merchant_monthly_revenue.xlsx
merchant_revenue_df = pd.read_excel("../data/raw/merchant_monthly_revenue.xlsx")

# copy merchant_revenue_df and call it df
df = merchant_revenue_df.copy()

# Ensure date column is datetime and set to month start
df["date"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp()

# filter df where merchant_id is "M001"
m_id = "M001"

# filter for merchant and set index
d = df[df["merchant_id"] == m_id].set_index("date")

# ensure revenue is float
y = d["revenue"].astype(float)

# separate data based on forecast horizon H
y_train, y_test = y.iloc[:-H], y.iloc[-H:]


### run forecasting methods
# SES
ses = SimpleExpSmoothing(y_train).fit(optimized=True)
ses_fc = ses.forecast(H)

# Holt -> trend
holt = ExponentialSmoothing(y_train, trend="add", seasonal=None).fit(optimized=True)
holt_fc = holt.forecast(H)

# Holt-Winters -> trend + seasonality
hw = ExponentialSmoothing(y_train, trend="add", seasonal="mul", seasonal_periods=12).fit(optimized=True)
hw_fc = hw.forecast(H)

# SARIMA
sarima = SARIMAX(
    y_train,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False)
sarima_fc = sarima.forecast(H)

# create forecasts df
preds = {
    "Naive": naive_forecast(y_train, H),
    "SeasonalNaive": seasonal_naive_forecast(y_train, H, 12),
    "SMA(3)": sma_forecast(y_train, H, 3),
    "WMA(1,2,3)": wma_forecast(y_train, H, (1, 2, 3)),
    "SES": ses_fc,
    "Holt": holt_fc,
    "HoltWinters": hw_fc,
    "SARIMA(111)(111)[12]": sarima_fc,
    # "Linear Regression": lin_fc,
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

# add actuals for comparison
forecasts_df.insert(0, "Actual", y_test.values)

# add merchant id column
forecasts_df.insert(0, "merchant_id", m_id)

### export to excel
forecasts_df.to_excel(Path("../data/transformed") / f"{m_id}_forecast.xlsx")
