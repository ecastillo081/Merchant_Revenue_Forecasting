from source.merchant_forecast import forecasts_df, y_test, y_train, m_id
from functions.metrics import mae, mape, smape, rmse, mase
import pandas as pd
from pathlib import Path

# drop merchant_id column from forecasts_df
forecasts_df.drop(columns=['merchant_id'], inplace=True, errors='ignore')

# calculate metrics for each model
metrics_rows = []
for col in forecasts_df.columns:
    if col == "Actual": continue
    yhat = forecasts_df[col]
    metrics_rows.append({
        "Model": col,
        "MAPE": mape(y_test, yhat),
        "sMAPE": smape(y_test, yhat),
        "MAE": mae(y_test, yhat),
        "RMSE": rmse(y_test, yhat),
        "MASE": mase(y_train, y_test, yhat, m=12)
    })
metrics_df = pd.DataFrame(metrics_rows).set_index("Model")

# save metrics to excel
metrics_df.to_excel(Path("../data/transformed") / f"{m_id}_metrics.xlsx")


### identify the best model
# choose best model metrics
chosen_metric = "MAPE"

def get_best_model(metrics_df, metric=chosen_metric):
    return metrics_df[metric].idxmin()

best_model = get_best_model(metrics_df, metric=chosen_metric)

print("Best model:", best_model)
