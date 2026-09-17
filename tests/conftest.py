"""Helpers for leakage and coverage tests."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_merchant_panel(
    merchant_id: str = "M001",
    start: str = "2020-01-01",
    periods: int = 60,
    seed: int = 1,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=periods, freq="MS")
    trend = np.linspace(80000, 100000, periods)
    season = 1 + 0.12 * np.sin(2 * np.pi * (dates.month - 1) / 12)
    noise = rng.normal(0, 1500, periods)
    promo = (rng.random(periods) < 0.15).astype(int)
    marketing = 6000 + 800 * np.sin(2 * np.pi * (dates.month - 1) / 12) + rng.normal(0, 200, periods)
    revenue = trend * season + 0.4 * marketing + 4000 * promo + noise
    macro = np.full(periods, 1.0)
    macro[-18:] = 1.03
    return pd.DataFrame(
        {
            "date": dates,
            "merchant_id": merchant_id,
            "revenue": revenue,
            "marketing_spend": np.clip(marketing, 1000, None),
            "promo_month": promo,
            "macro_index": macro,
        }
    )
