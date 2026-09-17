"""Merchant-window evaluation with origin-safe forecasts and coverage tracking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from functions.config import H, MIN_TRAIN_MONTHS, MODEL_NAMES
from functions.forecast_methods import (
    driver_scenario_regression_forecast,
    forecast_univariate,
    is_univariate,
)
from functions.metrics import mae, mape, mase, rmse, smape
from functions.planning_assumptions import build_driver_assumptions, forecast_index_from_cutoff


@dataclass
class WindowSpec:
    window_id: str
    cutoff: pd.Timestamp
    label: str = ""


def prepare_merchant_frame(df_merchant: pd.DataFrame) -> pd.DataFrame:
    g = df_merchant.copy()
    g["date"] = pd.to_datetime(g["date"]).dt.to_period("M").dt.to_timestamp()
    g = g.sort_values("date").drop_duplicates("date").set_index("date").asfreq("MS")
    if g["revenue"].isna().any():
        missing = g.index[g["revenue"].isna()].strftime("%Y-%m").tolist()
        raise ValueError(f"Missing monthly revenue at {missing}")
    return g


def split_origin(
    merchant_frame: pd.DataFrame,
    cutoff: pd.Timestamp,
    h: int = H,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = pd.Timestamp(cutoff)
    train = merchant_frame.loc[merchant_frame.index <= cutoff].copy()
    holdout_index = forecast_index_from_cutoff(cutoff, h)
    holdout = merchant_frame.reindex(holdout_index)
    if holdout["revenue"].isna().any():
        missing = holdout.index[holdout["revenue"].isna()].strftime("%Y-%m").tolist()
        raise ValueError(f"Holdout actuals missing at {missing}")
    if train.empty:
        raise ValueError("No training rows at or before the cutoff.")
    if train.index.max() > cutoff:
        raise ValueError("Training frame includes dates after the cutoff.")
    return train, holdout


def model_is_eligible(model_name: str, n_train_months: int) -> bool:
    return n_train_months >= MIN_TRAIN_MONTHS[model_name]


def forecast_one_model(
    model_name: str,
    train: pd.DataFrame,
    future_assumptions: pd.DataFrame | None = None,
    h: int = H,
) -> np.ndarray:
    y_train = train["revenue"].astype(float).asfreq("MS")
    if is_univariate(model_name):
        return forecast_univariate(model_name, y_train, h)
    if future_assumptions is None:
        raise ValueError("Driver Scenario Regression requires explicit future assumptions.")
    return driver_scenario_regression_forecast(train, future_assumptions, h=h)


def evaluate_merchant_window(
    df_merchant: pd.DataFrame,
    merchant_id: str,
    window: WindowSpec,
    h: int = H,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Return (metric_rows, forecast_rows, coverage_rows) for one merchant-window.

    Forecasts are generated from training data and origin-safe assumptions only.
    Holdout actuals are used solely for scoring after forecasts are produced.
    """
    frame = prepare_merchant_frame(df_merchant)
    train, holdout = split_origin(frame, window.cutoff, h=h)
    n_train = int(len(train))
    y_test = holdout["revenue"].astype(float)
    assumptions = build_driver_assumptions(train, holdout.index)

    metric_rows: list[dict] = []
    forecast_rows: list[dict] = []
    coverage_rows: list[dict] = []

    for model_name in MODEL_NAMES:
        coverage = {
            "window": window.window_id,
            "merchant_id": merchant_id,
            "model": model_name,
            "n_train_months": n_train,
            "min_train_months": MIN_TRAIN_MONTHS[model_name],
            "status": "success",
            "detail": "",
        }
        if not model_is_eligible(model_name, n_train):
            coverage["status"] = "not_eligible"
            coverage["detail"] = (
                f"Training history is {n_train} months; "
                f"{model_name} requires {MIN_TRAIN_MONTHS[model_name]}."
            )
            coverage_rows.append(coverage)
            continue
        try:
            yhat = np.asarray(
                forecast_one_model(model_name, train, assumptions, h=h),
                dtype=float,
            ).ravel()
            if yhat.shape[0] != h:
                raise ValueError(f"expected {h} steps, got {yhat.shape[0]}")
            if np.isnan(yhat).any():
                raise ValueError("forecast contains NaN")
        except Exception as exc:  # noqa: BLE001 - convert to coverage failure
            coverage["status"] = "failed"
            coverage["detail"] = f"{type(exc).__name__}: {exc}"
            coverage_rows.append(coverage)
            continue

        coverage_rows.append(coverage)
        metric_rows.append(
            {
                "window": window.window_id,
                "merchant_id": merchant_id,
                "model": model_name,
                "MAPE": mape(y_test, yhat),
                "sMAPE": smape(y_test, yhat),
                "MAE": mae(y_test, yhat),
                "RMSE": rmse(y_test, yhat),
                "MASE": mase(train["revenue"], y_test, yhat, m=12),
            }
        )
        for dt, actual, pred in zip(holdout.index, y_test.to_numpy(), yhat):
            forecast_rows.append(
                {
                    "window": window.window_id,
                    "merchant_id": merchant_id,
                    "date": pd.Timestamp(dt),
                    "actual": float(actual),
                    "model": model_name,
                    "forecast": float(pred),
                }
            )

    return metric_rows, forecast_rows, coverage_rows
