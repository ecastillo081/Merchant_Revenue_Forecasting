import sys
from itertools import cycle
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.single_merchant_forecast import forecasts_df, y_train, m_id
from source.single_merchant_forecast_metrics import best_model

FIGURES_DIR = ROOT / "figures"

graph_df = forecasts_df.copy()
graph_df.drop(columns=["merchant_id"], inplace=True, errors="ignore")


def plot_train_test_forecasts(y_train, graph_df, title="Actual vs Forecast", save_path=None):
    plt.figure(figsize=(12, 6))

    plt.plot(y_train.index, y_train, label="Train", color="gray", linewidth=1.5)
    plt.plot(graph_df.index, graph_df["Actual"], label="Actual", linewidth=2.5, color="black")

    color_cycle = cycle(plt.cm.tab10.colors)

    for col in graph_df.columns:
        if col == "Actual":
            continue
        lw = 3 if col == best_model else 1.8
        style = "--"
        color = "darkorange" if col == best_model else next(color_cycle)
        plt.plot(graph_df.index, graph_df[col], style, label=col, linewidth=lw, color=color)

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Revenue")
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter("${x:,.0f}"))
    plt.axvspan(graph_df.index[0], graph_df.index[-1], color="lightgrey", alpha=0.15)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()


FIGURES_DIR.mkdir(parents=True, exist_ok=True)
plot_train_test_forecasts(
    y_train,
    graph_df,
    title=f"{m_id} – Actual vs Forecast",
    save_path=FIGURES_DIR / f"{m_id}_actual_vs_forecast.png",
)
print(f"Saved {FIGURES_DIR / f'{m_id}_actual_vs_forecast.png'}")
