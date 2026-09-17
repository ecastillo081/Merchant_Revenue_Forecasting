"""Shared configuration for the merchant revenue forecasting case."""

from __future__ import annotations

H = 12
PRIMARY_METRIC = "MAPE"
PRIMARY_WINDOW = "2024"
EXPECTED_N_MERCHANTS = 50
BASELINE_MODEL = "Seasonal Naive"
DATA_START = "2020-01-01"

MODEL_NAMES = [
    "Naive",
    "Seasonal Naive",
    "SMA(3)",
    "WMA(1,2,3)",
    "SES",
    "Holt",
    "Holt-Winters",
    "SARIMA",
    "Driver Scenario Regression",
]

# Minimum training months required before a method is eligible for a window.
# SARIMA and Driver Scenario Regression need longer history: seasonal ARIMA with
# period 12 is unstable on 24 months, and the driver model drops 12 lag rows
# before fitting month dummies.
MIN_TRAIN_MONTHS = {
    "Naive": 1,
    "Seasonal Naive": 12,
    "SMA(3)": 3,
    "WMA(1,2,3)": 3,
    "SES": 2,
    "Holt": 3,
    "Holt-Winters": 24,
    "SARIMA": 36,
    "Driver Scenario Regression": 36,
}

EVALUATION_WINDOWS = [
    {
        "window_id": "2022",
        "cutoff": "2021-12-01",
        "label": "Forecast 2022 using data through December 2021",
    },
    {
        "window_id": "2023",
        "cutoff": "2022-12-01",
        "label": "Forecast 2023 using data through December 2022",
    },
    {
        "window_id": "2024",
        "cutoff": "2023-12-01",
        "label": "Forecast 2024 using data through December 2023",
    },
]

# Publication colors aligned to efrainfinance.com
COLOR_SELECTED = "#1a4554"
COLOR_BASELINE = "#c05621"
COLOR_OTHER = "#8b8680"
COLOR_INK = "#15202b"
COLOR_MUTED = "#4c5660"
COLOR_LINE = "#d6d2c8"
COLOR_HOLDOUT = "#eceae4"
COLOR_ACTUAL = "#15202b"
COLOR_PAPER = "#ffffff"
