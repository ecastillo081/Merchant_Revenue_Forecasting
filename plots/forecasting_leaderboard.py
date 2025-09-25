import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from source.all_merchants import chosen_metric
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# leaderboard file
leaderboard = pd.read_excel("../data/transformed/leaderboard.xlsx")
leaderboard_save_path = Path("../figures") / f"leaderboard_{chosen_metric}.png"
boxplot_save_path = Path("../figures") / f"boxplot_{chosen_metric}.png"


### bar plot of average metric by model
model_summary = (
    leaderboard.groupby("Model")[chosen_metric]
    .mean()
    .sort_values()
    .reset_index()
)

ax = model_summary.plot(
    kind="bar",
    x="Model",
    y=chosen_metric,
    legend=False,
    figsize=(9,6),
    color="skyblue",
    edgecolor="black"
)

ax.set_ylabel(f"Average {chosen_metric} (%)")
ax.set_xlabel("")
ax.set_title(f"Average Forecast Error Across All Merchants ({chosen_metric})")
plt.xticks(rotation=45, ha="right")

# add labels on top of each bar
for p in ax.patches:
    value = p.get_height()
    ax.annotate(
        f"{value:.1f}%",                   # format to 1 decimal with %
        (p.get_x() + p.get_width() / 2., value),
        ha='center', va='bottom',
        fontsize=9, color="black", xytext=(0,3), textcoords="offset points"
    )

plt.tight_layout()
plt.savefig(leaderboard_save_path, dpi=150)
plt.show()


### box plot with legend
models = [m for m in leaderboard["Model"].unique()]
data = [leaderboard.loc[leaderboard["Model"]==m, chosen_metric].values for m in models]

plt.figure(figsize=(10,5))
box = plt.boxplot(
    [leaderboard.loc[leaderboard["Model"]==m, chosen_metric].values
     for m in leaderboard["Model"].unique()],
    labels=leaderboard["Model"].unique(),
    showmeans=True,
    meanprops={"marker":"^","markerfacecolor":"green","markeredgecolor":"black"}
)

plt.ylabel(f"{chosen_metric} (%)")
plt.title("Forecast Accuracy Distribution by Model")
plt.xticks(rotation=45, ha="right")

# legend
box_patch = mpatches.Patch(facecolor="lightgray", edgecolor="black", label="25th–75th percentile (IQR)")
median_line = mlines.Line2D([], [], color="orange", label="Median")
mean_marker = mlines.Line2D([], [], color="green", marker="^", linestyle="None", markersize=8, label="Mean")
whisker_line = mlines.Line2D([], [], color="black", linestyle="-", label="Whiskers (range)")
outlier_marker = mlines.Line2D([], [], color="black", marker="o", linestyle="None", markersize=4, label="Outliers")

plt.legend(
    handles=[box_patch, median_line, mean_marker, whisker_line, outlier_marker],
    loc="upper right", fontsize=9, frameon=True
)

plt.tight_layout()
plt.savefig(boxplot_save_path, dpi=150)
plt.show()
