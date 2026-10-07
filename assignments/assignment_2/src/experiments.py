"""Run and analyse Assignment 2 experiments."""

# cSpell:ignore generalisation

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from config import CONDITIONS, FINAL_SEEDS, N_TEST_POSITIONS, RESULTS_DIR, TEST_RADIUS
from EA import mutation_parameters, run_ea, run_random_search
from eval import Evaluator

Row = dict[str, Any]


def save_csv(path: Path, rows: list[Row]) -> None:
    """Save list of dictionaries to CSV."""

    if not rows:
        return

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(condition: str, seed: int) -> Row:
    """Run one experiment and save results."""

    print()
    print("=" * 60)
    print(f"{condition.upper()} | seed {seed}")
    print("=" * 60)

    if condition == "random":
        result = run_random_search(seed)
    else:
        result = run_ea(condition, seed)

    evaluator = Evaluator()
    test_distances = evaluator.evaluate_generalisation(result["best_genotype"], N_TEST_POSITIONS, TEST_RADIUS)

    final_result: Row = {
        "condition": condition,
        "seed": seed,
        "training_fitness": result["best_fitness"],
        "test_mean": float(np.mean(test_distances)),
        "test_std": float(np.std(test_distances))
    }

    run_dir = RESULTS_DIR / condition / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    save_csv(run_dir / "history.csv", result["history"])
    save_csv(run_dir / "final.csv", [final_result])
    np.save(run_dir / "best_genotype.npy", result["best_genotype"])

    return final_result


def load_history(condition: str, seed: int) -> list[dict[str, str]]:
    """Load one history CSV."""

    path = RESULTS_DIR / condition / f"seed_{seed}" / "history.csv"

    if not path.exists():
        return []

    with open(path, encoding="utf-8") as file:
        return list(csv.DictReader(file))


def load_final_results() -> list[Row]:
    """Load completed final results."""

    results: list[Row] = []

    for condition in (*CONDITIONS, "random"):
        for seed in FINAL_SEEDS:
            path = RESULTS_DIR / condition / f"seed_{seed}" / "final.csv"

            if not path.exists():
                continue

            with open(path, encoding="utf-8") as file:
                row = next(csv.DictReader(file))

            results.append({
                "condition": condition,
                "seed": int(row["seed"]),
                "test_mean": float(row["test_mean"])
            })

    return results


def plot_learning_curves() -> None:
    """Plot mean and standard deviation over runs."""

    plt.figure(figsize=(8, 5))

    for condition in CONDITIONS:
        runs = []

        for seed in FINAL_SEEDS:
            rows = load_history(condition, seed)

            if not rows:
                continue

            best = [float(row["best"]) for row in rows]
            runs.append(best)

        if not runs:
            continue

        runs_array = np.asarray(runs)
        mean = np.mean(runs_array, axis=0)
        std = np.std(runs_array, axis=0)
        generations = np.arange(1, len(mean) + 1)

        plt.plot(generations, mean, label=condition)
        plt.fill_between(generations, mean - std, mean + std, alpha=0.2)

    plt.xlabel("Generation")
    plt.ylabel("Best fitness")
    plt.title("Evolutionary performance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "learning_curves.png", dpi=300)
    plt.close()


def plot_changed_weights() -> None:
    """Plot average number of changed weights."""

    plt.figure(figsize=(8, 5))

    for condition in CONDITIONS:
        runs = []

        for seed in FINAL_SEEDS:
            rows = load_history(condition, seed)

            if not rows:
                continue

            values = [float(row["mean_changed_weights"]) for row in rows]
            runs.append(values)

        if not runs:
            continue

        runs_array = np.asarray(runs)
        mean = np.mean(runs_array, axis=0)
        generations = np.arange(1, len(mean) + 1)

        plt.plot(generations, mean, label=condition)

    plt.xlabel("Generation")
    plt.ylabel("Mean changed weights")
    plt.title("Number of changed weights")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "changed_weights.png", dpi=300)
    plt.close()


def plot_improvement() -> None:
    """Plot average improvement of successful mutations."""

    plt.figure(figsize=(8, 5))

    for condition in CONDITIONS:
        runs = []

        for seed in FINAL_SEEDS:
            rows = load_history(condition, seed)

            if not rows:
                continue

            values = [float(row["mean_improvement"]) for row in rows]
            runs.append(values)

        if not runs:
            continue

        runs_array = np.asarray(runs)
        mean = np.mean(runs_array, axis=0)
        generations = np.arange(1, len(mean) + 1)

        plt.plot(generations, mean, label=condition)

    plt.xlabel("Generation")
    plt.ylabel("Mean fitness improvement")
    plt.title("Magnitude of successful improvements")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "improvement.png", dpi=300)
    plt.close()


def plot_final_scores() -> None:
    """Boxplot of final generalisation performance."""

    results = load_final_results()
    labels = []
    values = []

    for condition in (*CONDITIONS, "random"):
        scores = [row["test_mean"] for row in results if row["condition"] == condition]

        if scores:
            labels.append(condition)
            values.append(scores)

    if not values:
        return

    plt.figure(figsize=(7, 5))
    plt.boxplot(values, tick_labels=labels)
    plt.ylabel("Mean final distance (m)")
    plt.title("Final controller performance")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "final_scores.png", dpi=300)
    plt.close()


def print_results() -> None:
    """Print final results."""

    results = load_final_results()

    print()
    print("FINAL RESULTS")
    print("=" * 60)

    for condition in (*CONDITIONS, "random"):
        scores = np.asarray([row["test_mean"] for row in results if row["condition"] == condition])

        if len(scores) == 0:
            continue

        print(
            f"{condition:12s} | "
            f"mean = {np.mean(scores):.4f} | "
            f"std = {np.std(scores):.4f}"
        )


def print_design() -> None:
    """Show NN and mutation parameters."""

    evaluator = Evaluator()

    print()
    print("EXPERIMENTAL DESIGN")
    print("=" * 60)
    print(f"NN inputs  : {evaluator.input_size}")
    print(f"NN outputs : {evaluator.output_size}")
    print(f"Weights    : {evaluator.num_weights}")
    print()

    for condition in CONDITIONS:
        p, sigma = mutation_parameters(condition, evaluator.num_weights)
        expected_changed = evaluator.num_weights * p

        print(
            f"{condition:12s} | "
            f"p = {p:.5f} | "
            f"sigma = {sigma:.5f} | "
            f"E[changed] = {expected_changed:.1f}"
        )


def analyse() -> None:
    """Create all result figures."""

    plot_learning_curves()
    plot_changed_weights()
    plot_improvement()
    plot_final_scores()
    print_results()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=(*CONDITIONS, "random"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--analyse", action="store_true")
    parser.add_argument("--design", action="store_true")

    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.design:
        print_design()

    elif args.all:
        for condition in (*CONDITIONS, "random"):
            for seed in FINAL_SEEDS:
                run_experiment(condition, seed)

        analyse()

    elif args.analyse:
        analyse()

    elif args.condition is not None and args.seed is not None:
        run_experiment(args.condition, args.seed)

    else:
        print("Specify --design, --all, --analyse, or --condition CONDITION --seed SEED.")


if __name__ == "__main__":
    main()