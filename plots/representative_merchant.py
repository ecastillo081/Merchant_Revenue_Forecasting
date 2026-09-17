"""Representative merchant actual-vs-forecast chart for the planning case."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import StrMethodFormatter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from functions.config import (
    BASELINE_MODEL,
    COLOR_ACTUAL,
    COLOR_BASELINE,
    COLOR_HOLDOUT,
    COLOR_INK,
    COLOR_MUTED,
    COLOR_SELECTED,
    PRIMARY_WINDOW,
)
from plots.style import apply_style, savefig

RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
RAW_PATH = ROOT / "data" / "raw" / "merchant_monthly_revenue.xlsx"


def main() -> None:
    claims = json.loads((RESULTS_DIR / "publication_claims.json").read_text(encoding="utf-8"))
    info = claims["representative_merchant"]
    merchant_id = info["merchant_id"]
    selected = claims["selected_model"]
    metrics = pd.read_csv(RESULTS_DIR / "merchant_model_metrics.csv")
    metrics["window"] = metrics["window"].astype(str)
    metrics["merchant_id"] = metrics["merchant_id"].astype(str)
    forecasts = pd.read_csv(RESULTS_DIR / "holdout_forecasts.csv", parse_dates=["date"])
    forecasts["window"] = forecasts["window"].astype(str)
    forecasts["merchant_id"] = forecasts["merchant_id"].astype(str)
    raw = pd.read_excel(RAW_PATH)
    raw["date"] = pd.to_datetime(raw["date"]).dt.to_period("M").dt.to_timestamp()
    actual = (
        raw.loc[raw["merchant_id"].astype(str) == merchant_id, ["date", "revenue"]]
        .sort_values("date")
        .set_index("date")["revenue"]
        .astype(float)
    )

    holdout = forecasts.loc[
        (forecasts["window"] == str(PRIMARY_WINDOW)) & (forecasts["merchant_id"] == merchant_id)
    ].copy()
    selected_fc = holdout.loc[holdout["model"] == selected].set_index("date")["forecast"]
    baseline_fc = holdout.loc[holdout["model"] == BASELINE_MODEL].set_index("date")["forecast"]
    selected_mape = float(
        metrics.loc[
            (metrics["window"] == str(PRIMARY_WINDOW))
            & (metrics["merchant_id"] == merchant_id)
            & (metrics["model"] == selected),
            "MAPE",
        ].iloc[0]
    )
    baseline_mape = float(
        metrics.loc[
            (metrics["window"] == str(PRIMARY_WINDOW))
            & (metrics["merchant_id"] == merchant_id)
            & (metrics["model"] == BASELINE_MODEL),
            "MAPE",
        ].iloc[0]
    )

    holdout_start = selected_fc.index.min()
    plot_start = holdout_start - pd.DateOffset(years=2)
    actual_plot = actual.loc[actual.index >= plot_start]
    train_plot = actual_plot.loc[actual_plot.index < holdout_start]
    actual_holdout = actual_plot.loc[actual_plot.index >= holdout_start]

    apply_style()
    fig, ax = plt.subplots(figsize=(10.6, 4.7))
    ax.axvspan(holdout_start, actual.index.max(), color=COLOR_HOLDOUT, alpha=0.95, zorder=0)
    ax.plot(train_plot.index, train_plot, color="#7a756c", linewidth=1.7, label="Actual (training)")
    ax.plot(actual_holdout.index, actual_holdout, color=COLOR_ACTUAL, linewidth=2.4, label="Actual (holdout)")
    ax.plot(
        selected_fc.index,
        selected_fc,
        color=COLOR_SELECTED,
        linewidth=2.2,
        linestyle="--",
        label=f"{selected} ({selected_mape:.2f}% MAPE)",
    )
    ax.plot(
        baseline_fc.index,
        baseline_fc,
        color=COLOR_BASELINE,
        linewidth=2.0,
        linestyle="--",
        label=f"{BASELINE_MODEL} ({baseline_mape:.2f}% MAPE)",
    )
    ax.set_title(f"{merchant_id}: Actual Revenue vs Planning Forecasts", loc="left", pad=12, fontweight=650)
    ax.text(
        0,
        1.02,
        (
            f"Representative merchant: {selected} MAPE of {info['merchant_mape']:.2f}% is closest "
            f"to the 50-merchant median of {info['median_mape']:.2f}%. Shaded region is the 2024 holdout."
        ),
        transform=ax.transAxes,
        color=COLOR_MUTED,
        fontsize=9.0,
    )
    ax.set_ylabel("Revenue")
    ax.yaxis.set_major_formatter(StrMethodFormatter("${x:,.0f}"))
    ax.legend(loc="upper left", fontsize=8.2, frameon=True, framealpha=0.94)
    save_path = FIGURES_DIR / "representative_merchant_forecast.png"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    savefig(save_path, fig)
    print(f"Saved {save_path}")


if __name__ == "__main__":
    main()
