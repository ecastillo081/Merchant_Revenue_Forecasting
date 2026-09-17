"""Planning-assumption construction for driver-based forecasts.

Assumptions are built only from information available at the forecast origin.
"""

from __future__ import annotations

import pandas as pd


def forecast_index_from_cutoff(cutoff: pd.Timestamp, h: int) -> pd.DatetimeIndex:
    cutoff = pd.Timestamp(cutoff)
    start = cutoff + pd.offsets.MonthBegin(1)
    return pd.date_range(start=start, periods=h, freq="MS")


def build_driver_assumptions(
    train_df: pd.DataFrame,
    forecast_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Construct origin-safe driver plans for the forecast horizon.

    Training data must already be indexed by month-start dates and must not
    include holdout-period rows.

    Assumptions:
    - marketing_spend: same calendar month in the prior year (seasonal baseline)
    - promo_month: same calendar month in the prior-year promotion calendar
    - macro_index: last observed training-period value, carried forward
    """
    if train_df.empty:
        raise ValueError("Training data are required to build driver assumptions.")

    train = train_df.sort_index()
    last_macro = float(train["macro_index"].iloc[-1])
    rows = []
    for dt in pd.DatetimeIndex(forecast_index):
        prior_year = dt - pd.DateOffset(years=1)
        if prior_year not in train.index:
            raise ValueError(
                f"Prior-year training observation missing for {dt.date()} "
                f"(looked up {prior_year.date()})."
            )
        rows.append(
            {
                "date": dt,
                "marketing_spend": float(train.loc[prior_year, "marketing_spend"]),
                "promo_month": int(train.loc[prior_year, "promo_month"]),
                "macro_index": last_macro,
            }
        )
    assumptions = pd.DataFrame(rows).set_index("date")
    return assumptions
