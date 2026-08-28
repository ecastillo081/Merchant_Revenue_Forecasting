import sys
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.all_merchants import chosen_metric

FIGURES_DIR = ROOT / "figures"
TRANSFORMED_DIR = ROOT / "data" / "transformed"

leaderboard = pd.read_excel(TRANSFORMED_DIR / "leaderboard.xlsx")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
leaderboard_save_path = FIGURES_DIR / f"leaderboard_{chosen_metric}.png"
boxplot_save_path = FIGURES_DIR / f"boxplot_{chosen_metric}.png"

### bar plot of average metric by model
model_summary = (
    leaderboard.groupby("Model")[chosen_metric].mean().sort_values().reset_index()
)

ax = model_summary.plot(
    kind="bar",
    x="Model",
    y=chosen_metric,
    legend=False,
    figsize=(9, 6),
    color="skyblue",
    edgecolor="black",
)

ax.set_ylabel(f"Average {chosen_metric} (%)")
ax.set_xlabel("")
ax.set_title(f"Average Forecast Error Across All Merchants ({chosen_metric})")
plt.xticks(rotation=45, ha="right")

for p in ax.patches:
    value = p.get_height()
    ax.annotate(
        f"{value:.1f}%",
        (p.get_x() + p.get_width() / 2.0, value),
        ha="center",
        va="bottom",
        fontsize=9,
        color="black",
        xytext=(0, 3),
        textcoords="offset points",
    )

plt.tight_layout()
plt.savefig(leaderboard_save_path, dpi=150)
plt.close()

### box plot with legend
plt.figure(figsize=(10, 5))
plt.boxplot(
    [
        leaderboard.loc[leaderboard["Model"] == m, chosen_metric].values
        for m in leaderboard["Model"].unique()
    ],
    tick_labels=leaderboard["Model"].unique(),
    showmeans=True,
    meanprops={"marker": "^", "markerfacecolor": "green", "markeredgecolor": "black"},
)

plt.ylabel(f"{chosen_metric} (%)")
plt.title("Forecast Accuracy Distribution by Model")
plt.xticks(rotation=45, ha="right")

box_patch = mpatches.Patch(
    facecolor="lightgray", edgecolor="black", label="25th–75th percentile (IQR)"
)
median_line = mlines.Line2D([], [], color="orange", label="Median")
mean_marker = mlines.Line2D(
    [], [], color="green", marker="^", linestyle="None", markersize=8, label="Mean"
)
whisker_line = mlines.Line2D([], [], color="black", linestyle="-", label="Whiskers (range)")
outlier_marker = mlines.Line2D(
    [], [], color="black", marker="o", linestyle="None", markersize=4, label="Outliers"
)

plt.legend(
    handles=[box_patch, median_line, mean_marker, whisker_line, outlier_marker],
    loc="upper right",
    fontsize=9,
    frameon=True,
)

plt.tight_layout()
plt.savefig(boxplot_save_path, dpi=150)
plt.close()

print(f"Saved {leaderboard_save_path}")
print(f"Saved {boxplot_save_path}")
