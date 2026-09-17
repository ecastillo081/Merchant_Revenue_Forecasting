"""Coverage, uniqueness, and committed-result reconciliation tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from functions.config import (
    BASELINE_MODEL,
    EVALUATION_WINDOWS,
    EXPECTED_N_MERCHANTS,
    MODEL_NAMES,
    PRIMARY_WINDOW,
)
from functions.evaluation import WindowSpec, evaluate_merchant_window
from functions.metrics import point_difference, relative_reduction
from source.all_merchants import validate_coverage
from tests.conftest import make_merchant_panel

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def _require_results():
    needed = [
        "model_performance_summary.csv",
        "merchant_model_metrics.csv",
        "best_model_counts.csv",
        "rolling_window_summary.csv",
        "coverage_report.csv",
        "publication_claims.json",
    ]
    missing = [name for name in needed if not (RESULTS / name).exists()]
    if missing:
        pytest.skip(f"Committed results not present yet: {missing}")


def test_validate_coverage_fails_on_model_failure():
    coverage = pd.DataFrame(
        [
            {
                "window": "2024",
                "merchant_id": "M001",
                "model": model,
                "status": "failed" if model == "Holt" else "success",
                "detail": "boom" if model == "Holt" else "",
            }
            for model in MODEL_NAMES
        ]
    )
    with pytest.raises(RuntimeError, match="model failures"):
        validate_coverage(coverage, ["M001"], ["2024"])


def test_evaluate_records_failure_instead_of_skipping(monkeypatch):
    from functions import evaluation as ev

    def boom(*args, **kwargs):
        raise RuntimeError("forced failure")

    monkeypatch.setattr(ev, "forecast_one_model", boom)
    panel = make_merchant_panel(periods=60)
    window = WindowSpec("2024", pd.Timestamp("2023-12-01"))
    metrics, forecasts, coverage = evaluate_merchant_window(panel, "M001", window)
    assert metrics == []
    assert forecasts == []
    eligible_failed = [
        row
        for row in coverage
        if row["status"] == "failed" and row["n_train_months"] >= row["min_train_months"]
    ]
    assert eligible_failed
    assert len(eligible_failed) == len(
        [r for r in coverage if r["n_train_months"] >= r["min_train_months"]]
    )


def test_committed_coverage_is_complete():
    _require_results()
    coverage = pd.read_csv(RESULTS / "coverage_report.csv")
    coverage["window"] = coverage["window"].astype(str)
    coverage["merchant_id"] = coverage["merchant_id"].astype(str)
    windows = [row["window_id"] for row in EVALUATION_WINDOWS]
    merchants = sorted(coverage["merchant_id"].astype(str).unique())
    assert len(merchants) == EXPECTED_N_MERCHANTS
    validate_coverage(coverage, merchants, windows)
    assert (coverage["status"] == "failed").sum() == 0
    assert set(coverage["model"]) == set(MODEL_NAMES)


def test_each_expected_combination_appears_once():
    _require_results()
    metrics = pd.read_csv(RESULTS / "merchant_model_metrics.csv")
    metrics["window"] = metrics["window"].astype(str)
    coverage = pd.read_csv(RESULTS / "coverage_report.csv")
    coverage["window"] = coverage["window"].astype(str)
    success = coverage.loc[coverage["status"] == "success", ["window", "merchant_id", "model"]]
    merged = success.merge(metrics, on=["window", "merchant_id", "model"], how="left")
    assert merged["MAPE"].notna().all()
    assert not metrics.duplicated(["window", "merchant_id", "model"]).any()
    primary = metrics.loc[metrics["window"] == PRIMARY_WINDOW]
    assert primary["merchant_id"].nunique() == EXPECTED_N_MERCHANTS
    assert set(primary["model"]) == set(MODEL_NAMES)


def test_headline_metrics_match_committed_csv():
    _require_results()
    summary = pd.read_csv(RESULTS / "model_performance_summary.csv")
    claims = json.loads((RESULTS / "publication_claims.json").read_text(encoding="utf-8"))
    selected = summary.iloc[0]
    baseline = summary.loc[summary["model"] == BASELINE_MODEL].iloc[0]
    assert claims["selected_model"] == selected["model"]
    assert claims["selected_mean_mape"] == pytest.approx(selected["mean_mape"], rel=0, abs=1e-10)
    assert claims["baseline_mean_mape"] == pytest.approx(baseline["mean_mape"], rel=0, abs=1e-10)
    pp = point_difference(baseline["mean_mape"], selected["mean_mape"])
    rel = relative_reduction(baseline["mean_mape"], selected["mean_mape"])
    assert claims["point_difference"] == pytest.approx(pp, rel=0, abs=1e-10)
    assert claims["relative_reduction_pct"] == pytest.approx(rel, rel=0, abs=1e-10)


def test_best_model_counts_reconcile_to_merchant_results():
    _require_results()
    metrics = pd.read_csv(RESULTS / "merchant_model_metrics.csv")
    metrics["window"] = metrics["window"].astype(str)
    counts = pd.read_csv(RESULTS / "best_model_counts.csv")
    primary = metrics.loc[metrics["window"] == PRIMARY_WINDOW].copy()
    primary["model"] = pd.Categorical(primary["model"], categories=MODEL_NAMES, ordered=True)
    winner = (
        primary.sort_values(["merchant_id", "MAPE", "model"])
        .groupby("merchant_id", as_index=False, observed=True)
        .first()
    )
    computed = winner["model"].value_counts()
    for _, row in counts.iterrows():
        assert int(row["best_model_count"]) == int(computed.get(row["model"], 0))
    assert int(counts["best_model_count"].sum()) == EXPECTED_N_MERCHANTS


def test_relative_improvement_math():
    assert point_difference(12.47, 8.64) == pytest.approx(3.83)
    assert relative_reduction(12.47, 8.64) == pytest.approx(30.7137129, rel=1e-6)
