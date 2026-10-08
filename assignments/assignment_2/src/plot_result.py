
"""Generate figures and summary from completed experiment CSV files."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path("__data__/assignment_2/results")
CONDITIONS = ["0.05", "0.20", "0.50", "random"]


def load_results() -> pd.DataFrame:
    files = [RESULTS_DIR / f"{condition}_results.csv" for condition in CONDITIONS]

    for file in files:
        if not file.exists():
            raise FileNotFoundError(f"Missing results: {file}")

    results = pd.concat([pd.read_csv(file) for file in files], ignore_index=True)
    results.to_csv(RESULTS_DIR / "all_results.csv", index=False)
    return results


def get_final(results: pd.DataFrame) -> pd.DataFrame:
    return results.sort_values("generation").groupby(
        ["method", "seed"], as_index=False
    ).tail(1)


def plot_convergence(results: pd.DataFrame) -> None:
    max_generation = int(results["generation"].max())
    generations = range(max_generation + 1)
    extended: list[pd.DataFrame] = []

    for (method, seed), group in results.groupby(["method", "seed"]):
        group = group.set_index("generation").reindex(generations)
        group["best_fitness"] = group["best_fitness"].ffill()

        group = group.assign(method=str(method), seed=seed)
        extended.append(group.reset_index())

    data = pd.concat(extended, ignore_index=True)

    fig, ax = plt.subplots(figsize=(9, 5))

    for method, group in data.groupby("method"):
        stats = group.groupby("generation")["best_fitness"].agg(["mean", "std"])
        stats["std"] = stats["std"].fillna(0)

        ax.plot(stats.index, stats["mean"], label=method)
        ax.fill_between(
            stats.index,
            stats["mean"] - stats["std"],
            stats["mean"] + stats["std"],
            alpha=0.2,
        )

    ax.set(xlabel="Generation", ylabel="Best distance to target",
           title="Convergence across independent seeds")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "convergence.png", dpi=300)
    plt.close(fig)


def plot_final_fitness(results: pd.DataFrame) -> None:
    final = get_final(results)
    methods = list(final["method"].unique())
    values = [
        final.loc[final["method"] == method, "best_fitness"].to_numpy()
        for method in methods
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(values, tick_labels=methods)
    ax.set(ylabel="Final best distance to target",
           title="Final fitness across independent seeds")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "final_fitness.png", dpi=300)
    plt.close(fig)


def make_summary(results: pd.DataFrame) -> None:
    final = get_final(results)

    summary = final.groupby("method")["best_fitness"].agg(
        ["count", "mean", "std", "median", "min", "max"]
    ).round(4)

    summary.to_csv(RESULTS_DIR / "final_summary.csv")
    print("\nFinal fitness summary (lower is better):")
    print(summary)

def plot_stopping_generations(results: pd.DataFrame) -> None:
    final = get_final(results)
    final = final[final["method"] != "Random search"].copy()

    final["plateau_reached"] = final["generation"] < 130

    summary = final.groupby("method").agg(
        mean_generation=("generation", "mean"),
        std_generation=("generation", "std"),
        min_generation=("generation", "min"),
        max_generation=("generation", "max"),
        plateau_count=("plateau_reached", "sum"),
        total_runs=("seed", "count"),
    ).round(2)

    summary.to_csv(RESULTS_DIR / "stopping_summary.csv")
    print("\nStopping generation summary:")
    print(summary)

    methods = list(final["method"].unique())
    values = [
        final.loc[final["method"] == method, "generation"].to_numpy()
        for method in methods
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(values, tick_labels=methods)
    ax.axhline(130, linestyle="--", color="gray", label="Maximum generations")
    ax.set(xlabel="Mutation strength", ylabel="Stopping generation",
           title="Generations until plateau")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "stopping_generations.png", dpi=300)
    plt.close(fig)



def main() -> None:
    results = load_results()
    make_summary(results)
    plot_convergence(results)
    plot_final_fitness(results)
    plot_stopping_generations(results)
    print(f"\nFigures saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
