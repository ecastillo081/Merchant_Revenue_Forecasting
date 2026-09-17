"""Optional diagnostic: origin-safe forecasts for a single merchant."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from functions.config import PRIMARY_WINDOW
from functions.evaluation import WindowSpec, evaluate_merchant_window
from source.all_merchants import RAW_PATH, window_specs

TRANSFORMED_DIR = ROOT / "data" / "transformed"


def run(merchant_id: str = "M001", window_id: str = PRIMARY_WINDOW) -> pd.DataFrame:
    df = pd.read_excel(RAW_PATH)
    df["date"] = pd.to_datetime(df["date"])
    windows = {w.window_id: w for w in window_specs()}
    if window_id not in windows:
        raise ValueError(f"Unknown window {window_id}")
    window: WindowSpec = windows[window_id]
    g = df.loc[df["merchant_id"].astype(str) == merchant_id].copy()
    _, forecast_rows, coverage = evaluate_merchant_window(g, merchant_id, window)
    failed = [row for row in coverage if row["status"] == "failed"]
    if failed:
        raise RuntimeError(f"Forecast failure for {merchant_id}: {failed}")
    forecasts = pd.DataFrame(forecast_rows)
    wide = forecasts.pivot(index="date", columns="model", values="forecast")
    wide.insert(0, "Actual", forecasts.drop_duplicates("date").set_index("date")["actual"])
    wide.insert(0, "merchant_id", merchant_id)
    TRANSFORMED_DIR.mkdir(parents=True, exist_ok=True)
    wide.to_excel(TRANSFORMED_DIR / f"{merchant_id}_forecast.xlsx")
    return wide


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Origin-safe single-merchant diagnostic")
    parser.add_argument("--merchant-id", default="M001")
    parser.add_argument("--window", default=PRIMARY_WINDOW)
    args = parser.parse_args()
    out = run(args.merchant_id, args.window)
    print(out.head())
