import matplotlib.pyplot as plt
import pandas as pd


def epidemic_figure(results_df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    results_df[["Susceptible", "Infected (Untreated)", "Hospitalized", "Recovered"]].plot(
        ax=axes[0, 0], title="Population States"
    )
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("People")

    axes[0, 1].plot(results_df.index, results_df["Total Hospital Occupancy"], label="Occupancy", linewidth=2)
    axes[0, 1].plot(
        results_df.index,
        results_df["Effective Hospital Capacity"],
        label="Capacity",
        linewidth=2,
        linestyle="--",
    )
    axes[0, 1].set_title("Hospital Load vs Capacity")
    axes[0, 1].legend()

    axes[1, 0].plot(results_df.index, results_df["Effective Transmission Rate"], color="darkred", linewidth=2)
    axes[1, 0].set_title("Effective Transmission Rate")
    axes[1, 0].set_xlabel("Step")

    axes[1, 1].plot(results_df.index, results_df["Active Policies"], color="darkgreen", linewidth=2)
    axes[1, 1].set_title("Number of Active Policies")
    axes[1, 1].set_xlabel("Step")

    for ax in axes.ravel():
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
