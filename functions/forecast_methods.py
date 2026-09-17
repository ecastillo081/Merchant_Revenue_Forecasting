"""Forecast helpers with a 12-month planning horizon.

All methods receive only data available at the forecast origin. The driver
model additionally receives explicit future planning assumptions; it never
reads holdout-period actuals.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from functions.config import H, MODEL_NAMES

MONTH_DUMMY_COLS = [f"m_{month}" for month in range(2, 13)]
DRIVER_FEATURE_COLS = [
    "marketing_spend",
    "promo_month",
    "macro_index",
    "rev_lag1",
    "rev_lag12",
] + MONTH_DUMMY_COLS


def naive_forecast(y_train: pd.Series, h: int = H) -> np.ndarray:
    return np.repeat(float(y_train.iloc[-1]), h)


def seasonal_naive_forecast(y_train: pd.Series, h: int = H, m: int = 12) -> np.ndarray:
    if len(y_train) < m:
        raise ValueError(f"Seasonal naive requires at least {m} training months.")
    return np.resize(y_train.iloc[-m:].to_numpy(dtype=float), h)


def sma_forecast(y_train: pd.Series, h: int = H, window: int = 3) -> np.ndarray:
    if len(y_train) < window:
        raise ValueError(f"SMA requires at least {window} training months.")
    last = float(y_train.rolling(window).mean().iloc[-1])
    if np.isnan(last):
        raise ValueError("SMA window produced a missing value at the origin.")
    return np.repeat(last, h)


def wma_forecast(y_train: pd.Series, h: int = H, weights: tuple[int, ...] = (1, 2, 3)) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    if len(y_train) < len(w):
        raise ValueError(f"WMA requires at least {len(w)} training months.")
    last = float((y_train.iloc[-len(w):].to_numpy(dtype=float) * w).sum() / w.sum())
    return np.repeat(last, h)


def ses_forecast(y_train: pd.Series, h: int = H) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        fitted = SimpleExpSmoothing(y_train).fit(optimized=True)
    return np.asarray(fitted.forecast(h), dtype=float)


def holt_forecast(y_train: pd.Series, h: int = H) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        fitted = ExponentialSmoothing(
            y_train,
            trend="add",
            seasonal=None,
        ).fit(optimized=True)
    return np.asarray(fitted.forecast(h), dtype=float)


def holt_winters_forecast(y_train: pd.Series, h: int = H, m: int = 12) -> np.ndarray:
    if len(y_train) < 2 * m:
        raise ValueError("Holt-Winters requires at least two full seasonal cycles.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        fitted = ExponentialSmoothing(
            y_train,
            trend="add",
            seasonal="mul",
            seasonal_periods=m,
        ).fit(optimized=True)
    return np.asarray(fitted.forecast(h), dtype=float)


def sarima_forecast(y_train: pd.Series, h: int = H) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", UserWarning)
        fitted = SARIMAX(
            y_train,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
    return np.asarray(fitted.forecast(h), dtype=float)


def month_dummy_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    months = pd.Series(pd.DatetimeIndex(index).month, index=index, name="month")
    dummies = pd.get_dummies(months, prefix="m")
    out = pd.DataFrame(0.0, index=index, columns=MONTH_DUMMY_COLS)
    for col in MONTH_DUMMY_COLS:
        if col in dummies.columns:
            out[col] = dummies[col].astype(float)
    # January is the omitted month (no m_1 column), matching drop_first month effects.
    return out


def build_training_design_matrix(train_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Create lag features after the training cutoff and drop incomplete rows.

    No backfill or other future-looking imputation is applied.
    """
    g = train_df.sort_index().copy()
    g["rev_lag1"] = g["revenue"].shift(1)
    g["rev_lag12"] = g["revenue"].shift(12)
    X = pd.concat(
        [
            g[["marketing_spend", "promo_month", "macro_index", "rev_lag1", "rev_lag12"]].astype(float),
            month_dummy_frame(g.index),
        ],
        axis=1,
    )
    X.index = g.index
    valid = X.dropna(axis=0, how="any")
    if valid.empty:
        raise ValueError("No training rows remain after dropping incomplete lag history.")
    y = g.loc[valid.index, "revenue"].astype(float)
    return valid[DRIVER_FEATURE_COLS], y


def future_feature_row(
    date: pd.Timestamp,
    assumptions_row: pd.Series,
    rev_lag1: float,
    rev_lag12: float,
) -> pd.Series:
    month = int(pd.Timestamp(date).month)
    values = {
        "marketing_spend": float(assumptions_row["marketing_spend"]),
        "promo_month": float(assumptions_row["promo_month"]),
        "macro_index": float(assumptions_row["macro_index"]),
        "rev_lag1": float(rev_lag1),
        "rev_lag12": float(rev_lag12),
    }
    for col in MONTH_DUMMY_COLS:
        dummy_month = int(col.split("_")[1])
        values[col] = 1.0 if month == dummy_month else 0.0
    return pd.Series(values, dtype=float)


def driver_scenario_regression_forecast(
    train_df: pd.DataFrame,
    future_assumptions: pd.DataFrame,
    h: int = H,
    return_features: bool = False,
) -> np.ndarray | tuple[np.ndarray, pd.DataFrame]:
    """Scenario-conditioned 12-month forecast using origin-safe planning inputs.

    This is not a pure univariate forecast. Future marketing spend, promotions,
    and the macro index must be supplied as planning assumptions constructed
    from training-period information. ``rev_lag1`` is recursive (predicted
    values after the first step). ``rev_lag12`` uses known history for a
    12-month horizon.
    """
    if train_df.empty:
        raise ValueError("Driver Scenario Regression requires training data.")
    if len(future_assumptions) < h:
        raise ValueError("Future assumptions do not cover the forecast horizon.")

    train = train_df.sort_index()
    X_train, y_train = build_training_design_matrix(train)
    model = LinearRegression().fit(X_train.to_numpy(dtype=float), y_train.to_numpy(dtype=float))

    history = train["revenue"].astype(float).copy()
    assumptions = future_assumptions.sort_index().iloc[:h]
    preds = []
    feature_rows = []
    for dt, assumption_row in assumptions.iterrows():
        lag1 = float(history.iloc[-1])
        lag12_date = pd.Timestamp(dt) - pd.DateOffset(years=1)
        if lag12_date not in history.index:
            raise ValueError(f"rev_lag12 history missing for {pd.Timestamp(dt).date()}.")
        lag12 = float(history.loc[lag12_date])
        row = future_feature_row(dt, assumption_row, lag1, lag12)
        yhat = float(model.predict(row.to_numpy(dtype=float).reshape(1, -1))[0])
        preds.append(yhat)
        feature_rows.append(row)
        history.loc[pd.Timestamp(dt)] = yhat

    yhat_arr = np.asarray(preds, dtype=float)
    if return_features:
        return yhat_arr, pd.DataFrame(feature_rows, index=assumptions.index)
    return yhat_arr


def forecast_univariate(model_name: str, y_train: pd.Series, h: int = H) -> np.ndarray:
    if model_name == "Naive":
        return naive_forecast(y_train, h)
    if model_name == "Seasonal Naive":
        return seasonal_naive_forecast(y_train, h, 12)
    if model_name == "SMA(3)":
        return sma_forecast(y_train, h, 3)
    if model_name == "WMA(1,2,3)":
        return wma_forecast(y_train, h, (1, 2, 3))
    if model_name == "SES":
        return ses_forecast(y_train, h)
    if model_name == "Holt":
        return holt_forecast(y_train, h)
    if model_name == "Holt-Winters":
        return holt_winters_forecast(y_train, h)
    if model_name == "SARIMA":
        return sarima_forecast(y_train, h)
    raise KeyError(f"Unknown univariate model: {model_name}")


def is_univariate(model_name: str) -> bool:
    return model_name != "Driver Scenario Regression"


def known_model_names() -> list[str]:
    return list(MODEL_NAMES)
