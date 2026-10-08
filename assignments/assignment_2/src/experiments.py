
"""Run one experimental configuration across multiple independent seeds."""

import argparse
from pathlib import Path

import pandas as pd

from EA import GENERATIONS, POP_SIZE, HistoryRow, run_ea, run_random_search


# --------------------------------------------------------------------------- #
# EXPERIMENTAL SETTINGS
# --------------------------------------------------------------------------- #

MUTATION_STRENGTHS = [0.05, 0.20, 0.50]
N_SEEDS = 20

RESULTS_DIR = Path("__data__") / "assignment_2" / "results_r2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

type ResultRow = dict[str, str | int | float]


# --------------------------------------------------------------------------- #
# STORE RESULTS
# --------------------------------------------------------------------------- #

def add_history(rows: list[ResultRow], history: list[HistoryRow],
                method: str, seed: int) -> None:
    """Add one experimental run to the results table."""

    for row in history:
        rows.append({
            "method": method,
            "seed": seed,
            "generation": int(row["generation"]),
            "best_fitness": float(row["best"]),
            "mean_fitness": float(row["mean"]),
        })


# --------------------------------------------------------------------------- #
# RUN EXPERIMENT
# --------------------------------------------------------------------------- #

def run_experiment(condition: str, start_seed: int = 0,
                   n_seeds: int = N_SEEDS) -> None:
    """Run one configuration over independent seeds."""

    rows: list[ResultRow] = []
    output = RESULTS_DIR / f"{condition}_results.csv"

    for seed in range(start_seed, start_seed + n_seeds):
        if condition == "random":
            print(f"Random search | seed={seed}")

            history = run_random_search(
                seed=seed,
                population_size=POP_SIZE,
                generations=GENERATIONS,
            )
            method = "Random search"

        else:
            sigma = float(condition)
            print(f"EA | sigma={sigma:.2f} | seed={seed}")

            history, _ = run_ea(
                seed=seed,
                sigma=sigma,
                population_size=POP_SIZE,
                generations=GENERATIONS,
            )
            method = f"EA sigma={sigma:.2f}"

        add_history(rows, history, method, seed)

        pd.DataFrame(rows).to_csv(output, index=False)

        print(f"Finished seed {seed} | "
              f"Best fitness: {float(history[-1]['best']):.4f} | "
              f"Generations: {int(history[-1]['generation'])}")

    print(f"\nResults saved to: {output}")


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "condition",
        choices=["0.05", "0.20", "0.50", "random"],
    )
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=N_SEEDS)
    args = parser.parse_args()

    if args.num_seeds < 1:
        parser.error("--num-seeds must be at least 1")

    print(f"Population size: {POP_SIZE}")
    print(f"Maximum generations: {GENERATIONS}")
    print(f"Condition: {args.condition}")
    print(f"Seeds: {args.start_seed} to "
          f"{args.start_seed + args.num_seeds - 1}\n")

    run_experiment(args.condition, args.start_seed, args.num_seeds)


if __name__ == "__main__":
    main()
