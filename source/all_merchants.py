"""Portfolio benchmark: origin-safe methods, rolling windows, committed results."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from functions.config import (
    BASELINE_MODEL,
    EVALUATION_WINDOWS,
    EXPECTED_N_MERCHANTS,
    MODEL_NAMES,
    PRIMARY_METRIC,
    PRIMARY_WINDOW,
)
from functions.evaluation import WindowSpec, evaluate_merchant_window
from functions.metrics import iqr, point_difference, relative_reduction

RAW_PATH = ROOT / "data" / "raw" / "merchant_monthly_revenue.xlsx"
RESULTS_DIR = ROOT / "results"

chosen_metric = PRIMARY_METRIC


def window_specs() -> list[WindowSpec]:
    return [
        WindowSpec(
            window_id=row["window_id"],
            cutoff=pd.Timestamp(row["cutoff"]),
            label=row["label"],
        )
        for row in EVALUATION_WINDOWS
    ]


def load_panel(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_excel(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def expected_merchant_ids(df: pd.DataFrame) -> list[str]:
    ids = sorted(df["merchant_id"].dropna().astype(str).unique())
    return ids


def best_model_by_group(metrics: pd.DataFrame) -> pd.DataFrame:
    """One best model per merchant-window. Ties break toward simpler methods."""
    ranked = metrics.copy()
    ranked["model"] = pd.Categorical(ranked["model"], categories=MODEL_NAMES, ordered=True)
    ranked = ranked.sort_values(["window", "merchant_id", PRIMARY_METRIC, "model"])
    best = ranked.drop_duplicates(["window", "merchant_id"], keep="first")
    return best[["window", "merchant_id", "model", PRIMARY_METRIC]].rename(
        columns={"model": "best_model", PRIMARY_METRIC: "best_mape"}
    )


def summarize_primary(metrics: pd.DataFrame, best: pd.DataFrame, window_id: str) -> pd.DataFrame:
    subset = metrics.loc[metrics["window"] == window_id].copy()
    best_subset = best.loc[best["window"] == window_id]
    counts = best_subset["best_model"].value_counts()
    n_merchants = subset["merchant_id"].nunique()
    rows = []
    for model_name, grp in subset.groupby("model"):
        mape_vals = grp[PRIMARY_METRIC]
        wins = int(counts.get(model_name, 0))
        rows.append(
            {
                "model": model_name,
                "window": window_id,
                "n_merchants": int(grp["merchant_id"].nunique()),
                "mean_mape": float(mape_vals.mean()),
                "median_mape": float(mape_vals.median()),
                "iqr_mape": iqr(mape_vals),
                "mean_mae": float(grp["MAE"].mean()),
                "mean_rmse": float(grp["RMSE"].mean()),
                "mean_smape": float(grp["sMAPE"].mean()),
                "mean_mase": float(grp["MASE"].mean()),
                "best_model_count": wins,
                "best_model_pct": 100.0 * wins / n_merchants if n_merchants else float("nan"),
            }
        )
    summary = pd.DataFrame(rows).sort_values("mean_mape").reset_index(drop=True)
    summary.insert(0, "rank", range(1, len(summary) + 1))
    return summary


def summarize_rolling(metrics: pd.DataFrame) -> pd.DataFrame:
    window_means = (
        metrics.groupby(["window", "model"], as_index=False)[PRIMARY_METRIC]
        .mean()
        .rename(columns={PRIMARY_METRIC: "mean_mape"})
    )
    n_windows = int(metrics["window"].nunique())
    coverage = (
        metrics.groupby("model")["window"]
        .nunique()
        .rename("n_windows")
        .reset_index()
    )
    wide = window_means.pivot(index="model", columns="window", values="mean_mape")
    wide.columns = [f"mean_mape_{col}" for col in wide.columns]
    wide = wide.reset_index().merge(coverage, on="model", how="left")
    common = wide["n_windows"] == n_windows
    window_cols = [c for c in wide.columns if c.startswith("mean_mape_")]
    wide["mean_mape_across_windows"] = wide.loc[common, window_cols].mean(axis=1)
    ranked = wide.loc[common].sort_values("mean_mape_across_windows").copy()
    rank_map = {model: i for i, model in enumerate(ranked["model"], start=1)}
    wide["rank_common_set"] = wide["model"].map(rank_map)
    wide["comparison_set"] = np.where(
        common,
        "models eligible in every rolling window",
        "excluded from like-for-like rolling rank; incomplete window coverage",
    )
    window_cols = [c for c in wide.columns if c.startswith("mean_mape_") and c != "mean_mape_across_windows"]
    cols = ["rank_common_set", "model", "mean_mape_across_windows", "comparison_set"] + window_cols
    return wide[cols].sort_values(["rank_common_set", "model"], na_position="last").reset_index(drop=True)
    window_means = (
        metrics.groupby(["window", "model"], as_index=False)[PRIMARY_METRIC]
        .mean()
        .rename(columns={PRIMARY_METRIC: "mean_mape"})
    )
    coverage = (
        metrics.groupby("model")["window"]
        .nunique()
        .rename("n_windows")
        .reset_index()
    )
    n_windows = metrics["window"].nunique()
    common_models = coverage.loc[coverage["n_windows"] == n_windows, "model"]
    overall = (
        window_means.loc[window_means["model"].isin(common_models)]
        .groupby("model", as_index=False)["mean_mape"]
        .mean()
        .rename(columns={"mean_mape": "mean_mape_across_windows"})
        .sort_values("mean_mape_across_windows")
        .reset_index(drop=True)
    )
    overall.insert(0, "rank_common_set", range(1, len(overall) + 1))
    overall["comparison_set"] = "models eligible in every rolling window"
    wide = window_means.pivot(index="model", columns="window", values="mean_mape")
    wide.columns = [f"mean_mape_{col}" for col in wide.columns]
    out = overall.merge(wide.reset_index(), on="model", how="left")
    # Also attach models missing a window so the limitation is visible.
    missing = coverage.loc[coverage["n_windows"] < n_windows, "model"]
    if len(missing):
        extra = wide.reset_index()
        extra = extra.loc[extra["model"].isin(missing)].copy()
        extra["rank_common_set"] = pd.NA
        extra["mean_mape_across_windows"] = pd.NA
        extra["comparison_set"] = "excluded from like-for-like rolling rank; incomplete window coverage"
        out = pd.concat([out, extra], ignore_index=True, sort=False)
    return out


def select_representative_merchant(metrics: pd.DataFrame, selected_model: str, window_id: str) -> dict:
    subset = metrics.loc[
        (metrics["window"] == window_id) & (metrics["model"] == selected_model)
    ].copy()
    median_mape = float(subset[PRIMARY_METRIC].median())
    subset["abs_gap"] = (subset[PRIMARY_METRIC] - median_mape).abs()
    subset = subset.sort_values(["abs_gap", "merchant_id"])
    row = subset.iloc[0]
    return {
        "merchant_id": str(row["merchant_id"]),
        "selected_model": selected_model,
        "window": window_id,
        "merchant_mape": float(row[PRIMARY_METRIC]),
        "median_mape": median_mape,
        "selection_rule": (
            f"Merchant whose {selected_model} MAPE is closest to the "
            f"{window_id} median merchant-level MAPE for that model."
        ),
    }


def build_publication_claims(
    summary: pd.DataFrame,
    rolling: pd.DataFrame,
    representative: dict,
    n_merchants: int,
) -> dict:
    selected = summary.iloc[0]
    selected_model = str(selected["model"])
    baseline = summary.loc[summary["model"] == BASELINE_MODEL].iloc[0]
    selected_mape = float(selected["mean_mape"])
    baseline_mape = float(baseline["mean_mape"])
    pp = point_difference(baseline_mape, selected_mape)
    rel = relative_reduction(baseline_mape, selected_mape)

    rolling_common = rolling.loc[rolling["rank_common_set"].notna()].copy()
    rolling_winner = None
    rolling_winner_mape = None
    if not rolling_common.empty:
        top = rolling_common.sort_values("rank_common_set").iloc[0]
        rolling_winner = str(top["model"])
        rolling_winner_mape = float(top["mean_mape_across_windows"])

    selected_rolling = rolling.loc[rolling["model"] == selected_model]
    selected_rolling_mape = (
        float(selected_rolling["mean_mape_across_windows"].iloc[0])
        if not selected_rolling.empty and pd.notna(selected_rolling["mean_mape_across_windows"].iloc[0])
        else None
    )

    complex_notes = []
    for model_name in ["SARIMA", "Driver Scenario Regression", "Holt-Winters"]:
        match = summary.loc[summary["model"] == model_name]
        if not match.empty:
            row = match.iloc[0]
            complex_notes.append(
                f"{model_name} had a mean merchant-level MAPE of {row['mean_mape']:.2f}%"
            )

    recommend_as_default = selected_model == rolling_winner if rolling_winner else True
    return {
        "primary_metric": "Unweighted mean of merchant-level MAPE",
        "primary_window": PRIMARY_WINDOW,
        "n_merchants": n_merchants,
        "selected_model": selected_model,
        "selected_mean_mape": selected_mape,
        "selected_median_mape": float(selected["median_mape"]),
        "selected_iqr_mape": float(selected["iqr_mape"]),
        "selected_best_count": int(selected["best_model_count"]),
        "selected_best_pct": float(selected["best_model_pct"]),
        "baseline_model": BASELINE_MODEL,
        "baseline_mean_mape": baseline_mape,
        "point_difference": pp,
        "relative_reduction_pct": rel,
        "rolling_common_winner": rolling_winner,
        "rolling_common_winner_mean_mape": rolling_winner_mape,
        "selected_rolling_mean_mape": selected_rolling_mape,
        "recommend_as_default": bool(recommend_as_default),
        "complex_model_notes": complex_notes,
        "representative_merchant": representative,
        "display": {
            "selected_mean_mape": f"{selected_mape:.2f}",
            "baseline_mean_mape": f"{baseline_mape:.2f}",
            "point_difference": f"{pp:.2f}",
            "relative_reduction_pct": f"{rel:.1f}",
            "selected_best_count": int(selected["best_model_count"]),
            "selected_best_pct": f"{selected['best_model_pct']:.0f}",
        },
    }


def write_results(
    metrics: pd.DataFrame,
    forecasts: pd.DataFrame,
    coverage: pd.DataFrame,
    results_dir: Path = RESULTS_DIR,
) -> dict:
    metrics = metrics.copy()
    forecasts = forecasts.copy()
    coverage = coverage.copy()
    metrics["window"] = metrics["window"].astype(str)
    forecasts["window"] = forecasts["window"].astype(str)
    coverage["window"] = coverage["window"].astype(str)
    results_dir.mkdir(parents=True, exist_ok=True)
    best = best_model_by_group(metrics)
    primary = summarize_primary(metrics, best, PRIMARY_WINDOW)
    rolling = summarize_rolling(metrics)
    selected_model = str(primary.iloc[0]["model"])
    representative = select_representative_merchant(metrics, selected_model, PRIMARY_WINDOW)
    n_merchants = int(
        metrics.loc[metrics["window"] == PRIMARY_WINDOW, "merchant_id"].nunique()
    )
    claims = build_publication_claims(primary, rolling, representative, n_merchants)

    best_counts = primary[
        ["model", "best_model_count", "best_model_pct", "n_merchants", "window"]
    ].copy()

    metrics.to_csv(results_dir / "merchant_model_metrics.csv", index=False)
    primary.to_csv(results_dir / "model_performance_summary.csv", index=False)
    best_counts.to_csv(results_dir / "best_model_counts.csv", index=False)
    rolling.to_csv(results_dir / "rolling_window_summary.csv", index=False)
    coverage.to_csv(results_dir / "coverage_report.csv", index=False)
    forecasts.to_csv(results_dir / "holdout_forecasts.csv", index=False)
    best.to_csv(results_dir / "best_model_by_merchant.csv", index=False)
    with open(results_dir / "publication_claims.json", "w", encoding="utf-8") as f:
        json.dump(claims, f, indent=2)

    return claims


def validate_coverage(
    coverage: pd.DataFrame,
    expected_ids: list[str],
    expected_windows: list[str],
) -> None:
    failed = coverage.loc[coverage["status"] == "failed"]
    if not failed.empty:
        preview = failed.head(10).to_dict(orient="records")
        raise RuntimeError(
            f"Pipeline coverage incomplete: {len(failed)} model failures. "
            f"Examples: {preview}"
        )
    if len(expected_ids) != EXPECTED_N_MERCHANTS:
        raise RuntimeError(
            f"Expected {EXPECTED_N_MERCHANTS} merchants in the dataset, found {len(expected_ids)}."
        )
    missing_ids = sorted(set(expected_ids) - set(coverage["merchant_id"].unique()))
    if missing_ids:
        raise RuntimeError(f"Merchants missing from coverage report: {missing_ids}")
    missing_windows = sorted(set(expected_windows) - set(coverage["window"].astype(str).unique()))
    if missing_windows:
        raise RuntimeError(f"Windows missing from coverage report: {missing_windows}")

    expected_rows = len(expected_ids) * len(expected_windows) * len(MODEL_NAMES)
    if len(coverage) != expected_rows:
        raise RuntimeError(
            f"Coverage report has {len(coverage)} rows; expected {expected_rows} "
            f"unique merchant/model/window combinations."
        )
    dupes = coverage.duplicated(["window", "merchant_id", "model"]).sum()
    if dupes:
        raise RuntimeError(f"Coverage report contains {dupes} duplicate combinations.")


def run_pipeline(df: pd.DataFrame | None = None) -> dict:
    panel = load_panel() if df is None else df.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    merchant_ids = expected_merchant_ids(panel)
    windows = window_specs()

    metric_rows: list[dict] = []
    forecast_rows: list[dict] = []
    coverage_rows: list[dict] = []

    for window in windows:
        for merchant_id in merchant_ids:
            g = panel.loc[panel["merchant_id"].astype(str) == merchant_id].copy()
            m_rows, f_rows, c_rows = evaluate_merchant_window(g, merchant_id, window)
            metric_rows.extend(m_rows)
            forecast_rows.extend(f_rows)
            coverage_rows.extend(c_rows)
            print(f"Done: {merchant_id} / {window.window_id}")

    metrics = pd.DataFrame(metric_rows)
    forecasts = pd.DataFrame(forecast_rows)
    coverage = pd.DataFrame(coverage_rows)
    validate_coverage(coverage, merchant_ids, [w.window_id for w in windows])
    claims = write_results(metrics, forecasts, coverage)
    print("Complete.")
    print(
        f"Selected {PRIMARY_WINDOW} model: {claims['selected_model']} "
        f"({claims['display']['selected_mean_mape']}% mean merchant-level MAPE)"
    )
    return claims


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
