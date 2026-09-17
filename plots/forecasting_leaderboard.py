"""Publication leaderboard and merchant-level error distribution charts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from functions.config import BASELINE_MODEL, COLOR_INK, COLOR_MUTED, PRIMARY_METRIC, PRIMARY_WINDOW
from plots.style import apply_style, model_color, savefig

RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"


def load_claims() -> dict:
    return json.loads((RESULTS_DIR / "publication_claims.json").read_text(encoding="utf-8"))


def load_primary_metrics() -> pd.DataFrame:
    metrics = pd.read_csv(RESULTS_DIR / "merchant_model_metrics.csv")
    metrics["window"] = metrics["window"].astype(str)
    return metrics.loc[metrics["window"] == str(PRIMARY_WINDOW)].copy()


def ordered_models(summary: pd.DataFrame) -> list[str]:
    return summary.sort_values("mean_mape")["model"].tolist()


def plot_leaderboard(summary: pd.DataFrame, claims: dict, save_path: Path) -> None:
    apply_style()
    summary = summary.sort_values("mean_mape").reset_index(drop=True)
    selected = claims["selected_model"]
    colors = [model_color(name, selected) for name in summary["model"]]

    fig, ax = plt.subplots(figsize=(10.6, 5.15))
    bars = ax.bar(
        summary["model"],
        summary["mean_mape"],
        color=colors,
        width=0.72,
        zorder=2,
    )
    ax.set_ylabel("Mean merchant-level MAPE (%)")
    ax.set_xlabel("")
    ax.set_title("Mean Merchant-Level Forecast Error by Model", loc="left", pad=12, fontweight=650)
    ax.text(
        0,
        1.02,
        "Unweighted MAPE across 50 merchants; 12-month 2024 holdout",
        transform=ax.transAxes,
        color=COLOR_MUTED,
        fontsize=9.5,
    )
    ax.yaxis.grid(True, linestyle=":", alpha=0.55, zorder=0)
    ax.set_axisbelow(True)
    plt.xticks(rotation=28, ha="right")
    ymax = summary["mean_mape"].max()
    ax.set_ylim(0, ymax * 1.18)

    for bar, value in zip(bars, summary["mean_mape"]):
        ax.annotate(
            f"{value:.2f}%",
            (bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=COLOR_INK,
            xytext=(0, 3),
            textcoords="offset points",
        )

    d = claims["display"]
    ax.text(
        0.01,
        0.97,
        (
            f"{selected} is {d['point_difference']} percentage points lower than "
            f"{BASELINE_MODEL}\n({d['relative_reduction_pct']}% relative reduction in mean MAPE)"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.7,
        color=COLOR_INK,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f5f4f1", "edgecolor": "#d6d2c8"},
    )
    savefig(save_path, fig)


def plot_boxplot(metrics: pd.DataFrame, model_order: list[str], selected: str, save_path: Path) -> None:
    apply_style()
    data = [metrics.loc[metrics["model"] == name, PRIMARY_METRIC].to_numpy() for name in model_order]
    fig, ax = plt.subplots(figsize=(10.6, 4.85))
    bp = ax.boxplot(
        data,
        tick_labels=model_order,
        patch_artist=True,
        showmeans=True,
        meanprops={
            "marker": "^",
            "markerfacecolor": "#2f6f4e",
            "markeredgecolor": COLOR_INK,
            "markersize": 7,
        },
        medianprops={"color": "#c05621", "linewidth": 1.6},
        whiskerprops={"color": COLOR_INK},
        capprops={"color": COLOR_INK},
        flierprops={"marker": "o", "markersize": 4, "markerfacecolor": COLOR_MUTED, "markeredgecolor": COLOR_INK},
        boxprops={"linewidth": 1},
    )
    for patch, name in zip(bp["boxes"], model_order):
        patch.set_facecolor(model_color(name, selected))
        patch.set_alpha(0.78)
        patch.set_edgecolor(COLOR_INK)

    ax.set_ylabel("Merchant-level MAPE (%)")
    ax.set_title("Merchant-Level Error Distribution by Model", loc="left", pad=10, fontweight=650)
    ax.text(
        0,
        1.02,
        "Same model order as the leaderboard; 12-month 2024 holdout",
        transform=ax.transAxes,
        color=COLOR_MUTED,
        fontsize=9.5,
    )
    plt.xticks(rotation=28, ha="right")
    ax.yaxis.grid(True, linestyle=":", alpha=0.55)
    ax.set_axisbelow(True)

    handles = [
        mpatches.Patch(facecolor="#8b8680", edgecolor=COLOR_INK, label="25th–75th percentile (IQR)"),
        mlines.Line2D([], [], color="#c05621", label="Median"),
        mlines.Line2D([], [], color="#2f6f4e", marker="^", linestyle="None", label="Mean"),
        mlines.Line2D([], [], color=COLOR_INK, label="Whiskers: 1.5× IQR"),
        mlines.Line2D([], [], color=COLOR_INK, marker="o", linestyle="None", markersize=4, label="Outliers"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8.4, frameon=True, framealpha=0.95)
    savefig(save_path, fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    claims = load_claims()
    summary = pd.read_csv(RESULTS_DIR / "model_performance_summary.csv")
    metrics = load_primary_metrics()
    order = ordered_models(summary)
    plot_leaderboard(summary, claims, FIGURES_DIR / "leaderboard_MAPE.png")
    plot_boxplot(metrics, order, claims["selected_model"], FIGURES_DIR / "boxplot_MAPE.png")
    print(f"Saved {FIGURES_DIR / 'leaderboard_MAPE.png'}")
    print(f"Saved {FIGURES_DIR / 'boxplot_MAPE.png'}")


if __name__ == "__main__":
    main()
