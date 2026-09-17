"""Forecast-origin leakage tests for Driver Scenario Regression and evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]

from functions.config import H, MODEL_NAMES
from functions.evaluation import (
    WindowSpec,
    evaluate_merchant_window,
    prepare_merchant_frame,
    split_origin,
)
from functions.forecast_methods import (
    build_training_design_matrix,
    driver_scenario_regression_forecast,
)
from functions.planning_assumptions import build_driver_assumptions
from tests.conftest import make_merchant_panel


def test_training_rows_never_after_cutoff():
    panel = make_merchant_panel()
    frame = prepare_merchant_frame(panel)
    cutoff = pd.Timestamp("2023-12-01")
    train, _ = split_origin(frame, cutoff, h=H)
    assert train.index.max() <= cutoff
    assert (train.index > cutoff).sum() == 0


def test_holdout_actual_revenue_not_used_for_lags():
    panel = make_merchant_panel()
    frame = prepare_merchant_frame(panel)
    cutoff = pd.Timestamp("2023-12-01")
    train, holdout = split_origin(frame, cutoff, h=H)
    assumptions = build_driver_assumptions(train, holdout.index)
    _, features = driver_scenario_regression_forecast(
        train, assumptions, h=H, return_features=True
    )
    holdout_revenue = set(np.round(holdout["revenue"].to_numpy(), 6))
    used_lag1 = set(np.round(features["rev_lag1"].to_numpy(), 6))
    used_lag12 = set(np.round(features["rev_lag12"].to_numpy(), 6))
    assert used_lag1.isdisjoint(holdout_revenue)
    assert used_lag12.isdisjoint(holdout_revenue)


def test_changing_holdout_actuals_does_not_change_forecast():
    panel = make_merchant_panel()
    window = WindowSpec("2024", pd.Timestamp("2023-12-01"))
    metrics_orig, forecasts_orig, coverage_orig = evaluate_merchant_window(
        panel, "M001", window
    )
    assert all(row["status"] in {"success", "not_eligible"} for row in coverage_orig)

    mutated = panel.copy()
    holdout_mask = pd.to_datetime(mutated["date"]) >= pd.Timestamp("2024-01-01")
    mutated.loc[holdout_mask, "revenue"] = mutated.loc[holdout_mask, "revenue"] * 12 + 99999
    mutated.loc[holdout_mask, "marketing_spend"] = 1.0
    mutated.loc[holdout_mask, "promo_month"] = 1
    mutated.loc[holdout_mask, "macro_index"] = 9.9

    _, forecasts_mut, _ = evaluate_merchant_window(mutated, "M001", window)
    orig = pd.DataFrame(forecasts_orig)
    mut = pd.DataFrame(forecasts_mut)
    merged = orig.merge(
        mut,
        on=["window", "merchant_id", "date", "model"],
        suffixes=("_orig", "_mut"),
    )
    np.testing.assert_allclose(merged["forecast_orig"], merged["forecast_mut"], rtol=0, atol=1e-10)
    assert not np.allclose(merged["actual_orig"], merged["actual_mut"])


def test_recursive_rev_lag1_uses_predictions_not_holdout():
    panel = make_merchant_panel()
    frame = prepare_merchant_frame(panel)
    cutoff = pd.Timestamp("2023-12-01")
    train, holdout = split_origin(frame, cutoff, h=H)
    assumptions = build_driver_assumptions(train, holdout.index)
    preds, features = driver_scenario_regression_forecast(
        train, assumptions, h=H, return_features=True
    )
    assert features.iloc[0]["rev_lag1"] == pytest.approx(float(train["revenue"].iloc[-1]))
    assert features.iloc[1]["rev_lag1"] == pytest.approx(float(preds[0]))
    for step in range(1, H):
        assert features.iloc[step]["rev_lag1"] == pytest.approx(float(preds[step - 1]))


def test_rev_lag12_uses_known_history_on_12_month_horizon():
    panel = make_merchant_panel()
    frame = prepare_merchant_frame(panel)
    cutoff = pd.Timestamp("2023-12-01")
    train, holdout = split_origin(frame, cutoff, h=H)
    assumptions = build_driver_assumptions(train, holdout.index)
    _, features = driver_scenario_regression_forecast(
        train, assumptions, h=H, return_features=True
    )
    for dt, row in features.iterrows():
        lag12_date = pd.Timestamp(dt) - pd.DateOffset(years=1)
        assert lag12_date in train.index
        assert row["rev_lag12"] == pytest.approx(float(train.loc[lag12_date, "revenue"]))


def test_no_future_looking_backfill_in_training_matrix():
    panel = make_merchant_panel()
    frame = prepare_merchant_frame(panel)
    train, _ = split_origin(frame, pd.Timestamp("2023-12-01"), h=H)
    X, y = build_training_design_matrix(train)
    assert X.index.min() == train.index[12]
    assert not X.isna().any().any()
    assert len(X) == len(train) - 12
    assert len(X) == len(y)
    source = (ROOT / "functions" / "forecast_methods.py").read_text(encoding="utf-8")
    assert "bfill" not in source
    assert "fillna(method" not in source


def test_driver_assumptions_come_from_training_only():
    panel = make_merchant_panel()
    frame = prepare_merchant_frame(panel)
    cutoff = pd.Timestamp("2023-12-01")
    train, holdout = split_origin(frame, cutoff, h=H)
    assumptions = build_driver_assumptions(train, holdout.index)
    last_macro = float(train["macro_index"].iloc[-1])
    for dt, row in assumptions.iterrows():
        prior = pd.Timestamp(dt) - pd.DateOffset(years=1)
        assert row["marketing_spend"] == pytest.approx(float(train.loc[prior, "marketing_spend"]))
        assert int(row["promo_month"]) == int(train.loc[prior, "promo_month"])
        assert row["macro_index"] == pytest.approx(last_macro)
        assert prior <= cutoff


def test_expected_model_names_are_complete():
    assert "Linear Regression" not in MODEL_NAMES
    assert "Driver Scenario Regression" in MODEL_NAMES
    assert len(MODEL_NAMES) == 9
