import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from functions.forecast_methods import H, naive_forecast, seasonal_naive_forecast, sma_forecast, wma_forecast, linear_regression_forecast
from functions.metrics import mae, mape, smape, rmse, mase

merchant_revenue_df = pd.read_excel("../data/raw/merchant_monthly_revenue.xlsx")

# best model metric
chosen_metric = "MAPE"

def forecasts_to_df(preds_dict, y_test_index):
    out = pd.DataFrame(index=y_test_index)
    for name, fc in preds_dict.items():
        if fc is None:
            continue
        fc = np.asarray(fc, dtype=float).ravel()
        if len(fc) != len(y_test_index):
            raise ValueError(f"{name}: expected {len(y_test_index)} steps, got {len(fc)}")
        out[name] = fc
    return out

def evaluate_one_merchant(df_merchant, m_id, chosen_metric=chosen_metric):
    g = df_merchant.sort_values("date").copy()
    g["date"] = pd.to_datetime(g["date"]).dt.to_period("M").dt.to_timestamp()
    g = g.set_index("date").asfreq("MS")
    y = g["revenue"].astype(float)

    # check to make sure at least 24 months of data or H + 6 months
    if len(y) < max(24, H + 6):
        return None, None, None

    y_train, y_test = y.iloc[:-H], y.iloc[-H:]

    # forecasting plots
    ses = SimpleExpSmoothing(y_train).fit(optimized=True)
    ses_fc = ses.forecast(H)

    holt = ExponentialSmoothing(y_train, trend="add", seasonal=None).fit(optimized=True)
    holt_fc = holt.forecast(H)

    hw = ExponentialSmoothing(y_train, trend="add", seasonal="mul", seasonal_periods=12).fit(optimized=True)
    hw_fc = hw.forecast(H)

    sarima = SARIMAX(
        y_train, order=(1,1,1), seasonal_order=(1,1,1,12),
        enforce_stationarity=False, enforce_invertibility=False
    ).fit(disp=False)
    sarima_fc = sarima.forecast(H)

    # linear regression
    lin_fc = linear_regression_forecast(df_merchant, H)

    preds = {
        "Naive": naive_forecast(y_train, H),
        "Seasonal Naive": seasonal_naive_forecast(y_train, H, 12),
        "SMA(3)": sma_forecast(y_train, H, 3),
        "WMA(1,2,3)": wma_forecast(y_train, H, (1,2,3)),
        "SES": ses_fc,
        "Holt": holt_fc,
        "Holt-Winters": hw_fc,
        "SARIMA": sarima_fc,
        "Linear Regression": lin_fc,
    }

    # forecasts df - per merchant
    fc_df = forecasts_to_df(preds, y_test.index)
    fc_df.insert(0, "Actual", y_test.values)

    # metrics df - per model
    rows = []
    for name, yhat in preds.items():
        rows.append({
            "merchant_id": m_id,
            "Model": name,
            "MAPE": mape(y_test, yhat),
            "sMAPE": smape(y_test, yhat),
            "MAE": mae(y_test, yhat),
            "RMSE": rmse(y_test, yhat),
            "MASE": mase(y_train, y_test, yhat, m=12)
        })
    metrics_df = pd.DataFrame(rows).set_index("Model")

    # choose best model
    best_model = metrics_df[chosen_metric].idxmin()

    # best model summary
    best_row = metrics_df.loc[[best_model]].copy()
    best_row.insert(1, "Best_By", chosen_metric)
    best_row = best_row.reset_index().rename(columns={"index":"Model"})

    return fc_df, metrics_df.reset_index(), best_row

def main():
    df = merchant_revenue_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    merchant_ids = df["merchant_id"].dropna().unique()
    all_metrics = []
    all_best = []

    for m_id in merchant_ids:
        g = df[df["merchant_id"] == m_id].copy()
        try:
            fc_df, metrics_df, best_row = evaluate_one_merchant(g, m_id)
            if metrics_df is not None:
                all_metrics.append(metrics_df.assign(merchant_id=m_id))
            if best_row is not None:
                all_best.append(best_row)
            print(f"Done: {m_id}")
        except Exception as e:
            print(f"Skip {m_id}: {e}")

    if all_metrics:
        leaderboard = pd.concat(all_metrics, ignore_index=True)
        leaderboard.to_excel("../data/transformed/leaderboard.xlsx", index=False)

    if all_best:
        best_all = pd.concat(all_best, ignore_index=True)
        best_all.to_excel("../data/transformed/best_model_by_merchant.xlsx", index=False)

    print("Complete.")

if __name__ == "__main__":
    main()
