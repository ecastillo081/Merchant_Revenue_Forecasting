import matplotlib.pyplot as plt
from source.merchant_forecast import forecasts_df, y_train, m_id
from source.merchant_forecast_metrics import best_model
from pathlib import Path
from matplotlib.ticker import StrMethodFormatter
from itertools import cycle

# copy forecasts_df and call it graph_df
graph_df = forecasts_df.copy()

# drop merchant_id column
graph_df.drop(columns=['merchant_id'], inplace=True, errors='ignore')

# prep plot graph comparing forecasts to actuals
def plot_train_test_forecasts(y_train, graph_df, title="Actual vs Forecast", save_path=None):
    plt.figure(figsize=(12, 6))

    # plot training data
    plt.plot(y_train.index, y_train, label="Train", color="gray", linewidth=1.5)

    # plot actuals
    plt.plot(graph_df.index, graph_df["Actual"], label="Actual", linewidth=2.5, color="black")

    # skip black & gray colors for visual clarity
    color_cycle = cycle(plt.cm.tab10.colors)

    # plot forecasts
    for col in graph_df.columns:
        if col == "Actual": continue
        lw = 3 if col == best_model else 1.8
        style = '--' if col == best_model else '--'
        color = 'darkorange' if col == best_model else next(color_cycle)
        plt.plot(graph_df.index, graph_df[col], style, label=col, linewidth=lw, color=color)

    # title and axis labels
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Revenue")

    # format y-axis as dollars with commas
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}'))

    plt.axvspan(graph_df.index[0], graph_df.index[-1], color='lightgrey', alpha=0.15)

    # legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.)
    plt.tight_layout()

    # save
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

# plot graph
plot_train_test_forecasts(
    y_train,
    graph_df,
    title=f"{m_id} – Actual vs Forecast",
    save_path=Path("../figures") / f"{m_id}_actual_vs_forecast.png"
)
