"""Run the mutation-strength experiment for Assignment 2."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from EA import (
    GENERATIONS,
    POP_SIZE,
    HistoryRow,
    run_ea,
    run_random_search,
)


# --------------------------------------------------------------------------- #
# EXPERIMENTAL SETTINGS
# --------------------------------------------------------------------------- #

# This is the ONLY thing that changes between EA conditions
MUTATION_STRENGTHS = [0.05, 0.20, 0.50]
SEEDS = list(range(5))


RESULTS_DIR = (
    Path("__data__")
    / "assignment_2"
    / "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


type ResultRow = dict[str, str | int | float]


# --------------------------------------------------------------------------- #
# STORE RESULTS
# --------------------------------------------------------------------------- #

def add_history(
    rows: list[ResultRow],
    history: list[HistoryRow],
    method: str,
    seed: int,
) -> None:
    """Add one experimental run to the results table."""

    for row in history:

        rows.append(
            {
                "method": method,
                "seed": seed,
                "generation": int(row["generation"]),
                "best_fitness": float(row["best"]),
                "mean_fitness": float(row["mean"]),
            }
        )


# --------------------------------------------------------------------------- #
# RUN EXPERIMENT
# --------------------------------------------------------------------------- #

def run_all_experiments() -> pd.DataFrame:
    """Run all EA settings and the random-search baseline."""

    rows: list[ResultRow] = []

    for sigma in MUTATION_STRENGTHS:
        for seed in SEEDS:
            print(f"EA | sigma={sigma:.2f} | seed={seed}")

            history, _ = run_ea(
                seed=seed,
                sigma=sigma,
                population_size=POP_SIZE,
                generations=GENERATIONS,
            )

            add_history(
                rows,
                history,
                method=f"EA sigma={sigma:.2f}",
                seed=seed,
            )

    for seed in SEEDS:
        print(f"Random search | seed={seed}")

        history = run_random_search(
            seed=seed,
            population_size=POP_SIZE,
            generations=GENERATIONS,
        )

        add_history(
            rows,
            history,
            method="Random search",
            seed=seed,
        )

    results = pd.DataFrame(rows)

    results.to_csv(
        RESULTS_DIR / "all_results.csv",
        index=False,
    )

    return results


# --------------------------------------------------------------------------- #
# SUMMARY
# --------------------------------------------------------------------------- #

def make_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize final fitness across the 10 seeds."""

    final = results[
        results["generation"] == GENERATIONS
    ]

    summary = (
        final
        .groupby("method")["best_fitness"]
        .agg(
            [
                "mean",
                "std",
                "median",
                "min",
                "max",
            ]
        )
        .reset_index()
        .sort_values("mean")
    )

    summary.to_csv(
        RESULTS_DIR / "final_summary.csv",
        index=False,
    )

    return summary


# --------------------------------------------------------------------------- #
# CONVERGENCE PLOT
# --------------------------------------------------------------------------- #

def plot_convergence(
    results: pd.DataFrame,
) -> None:
    """Plot mean best fitness +/- one standard deviation."""

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    for method, data in results.groupby(
        "method",
        sort=False,
    ):

        stats = (
            data
            .groupby("generation")["best_fitness"]
            .agg(["mean", "std"])
            .reset_index()
        )

        ax.plot(
            stats["generation"],
            stats["mean"],
            label=method,
        )

        ax.fill_between(
            stats["generation"],
            stats["mean"] - stats["std"],
            stats["mean"] + stats["std"],
            alpha=0.2,
        )

    ax.set_xlabel("Generation")

    ax.set_ylabel(
        "Best distance to target"
    )

    ax.set_title(
        "Convergence over 10 independent seeds"
    )

    ax.legend()

    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()

    fig.savefig(
        RESULTS_DIR / "convergence.png",
        dpi=300,
    )

    plt.close(fig)


# --------------------------------------------------------------------------- #
# FINAL FITNESS PLOT
# --------------------------------------------------------------------------- #

def plot_final_fitness(
    results: pd.DataFrame,
) -> None:
    """Compare final performance across the 10 seeds."""

    final = results[
        results["generation"] == GENERATIONS
    ]

    methods = list(
        final["method"].drop_duplicates()
    )

    values = [
        final.loc[
            final["method"] == method,
            "best_fitness",
        ].to_numpy()
        for method in methods
    ]

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.boxplot(
        values,
        tick_labels=methods,
    )

    ax.set_ylabel(
        "Final best distance to target"
    )

    ax.set_title(
        "Final performance over 10 independent seeds"
    )

    ax.tick_params(
        axis="x",
        rotation=20,
    )

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    fig.tight_layout()

    fig.savefig(
        RESULTS_DIR / "final_fitness.png",
        dpi=300,
    )

    plt.close(fig)


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #

def main() -> None:
    print(f"Population size: {POP_SIZE}")
    print(f"Generations: {GENERATIONS}")
    print(f"Seeds per setting: {len(SEEDS)}")
    print(f"Mutation strengths: {MUTATION_STRENGTHS}")

    results = run_all_experiments()

    summary = make_summary(results)

    plot_convergence(results)
    plot_final_fitness(results)

    print("\nFinal summary (lower is better):")
    print(summary.to_string(index=False))

    print(f"\nResults saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()