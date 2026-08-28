import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from functions.metrics import mae, mape, smape, rmse, mase
from source.single_merchant_forecast import forecasts_df, y_test, y_train, m_id

TRANSFORMED_DIR = ROOT / "data" / "transformed"

forecasts_df.drop(columns=["merchant_id"], inplace=True, errors="ignore")

metrics_rows = []
for col in forecasts_df.columns:
    if col == "Actual":
        continue
    yhat = forecasts_df[col]
    metrics_rows.append(
        {
            "Model": col,
            "MAPE": mape(y_test, yhat),
            "sMAPE": smape(y_test, yhat),
            "MAE": mae(y_test, yhat),
            "RMSE": rmse(y_test, yhat),
            "MASE": mase(y_train, y_test, yhat, m=12),
        }
    )
metrics_df = pd.DataFrame(metrics_rows).set_index("Model")

TRANSFORMED_DIR.mkdir(parents=True, exist_ok=True)
metrics_df.to_excel(TRANSFORMED_DIR / f"{m_id}_metrics.xlsx")

chosen_metric = "MAPE"


def get_best_model(metrics_df, metric=chosen_metric):
    return metrics_df[metric].idxmin()


best_model = get_best_model(metrics_df, metric=chosen_metric)

print("Best model:", best_model)
