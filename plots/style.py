"""Shared matplotlib style for publication charts."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

from functions.config import (
    BASELINE_MODEL,
    COLOR_BASELINE,
    COLOR_INK,
    COLOR_MUTED,
    COLOR_OTHER,
    COLOR_SELECTED,
)


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Calibri", "DejaVu Sans", "Arial"],
            "axes.edgecolor": "#d6d2c8",
            "axes.labelcolor": COLOR_INK,
            "axes.titlecolor": COLOR_INK,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "xtick.color": COLOR_MUTED,
            "ytick.color": COLOR_MUTED,
            "text.color": COLOR_INK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
        }
    )


def model_color(model_name: str, selected_model: str) -> str:
    if model_name == selected_model:
        return COLOR_SELECTED
    if model_name == BASELINE_MODEL:
        return COLOR_BASELINE
    return COLOR_OTHER


def savefig(path, fig=None, dpi: int = 180) -> None:
    fig = fig or plt.gcf()
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
