"""Run and analyse Assignment 2 experiments."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from EA import RunResult, mutation_parameters, run_ea, run_random_search
from eval import Evaluator
from config import (
    CONDITIONS,
    FINAL_SEEDS,
    N_TEST_POSITIONS,
    RESULTS_DIR,
    TEST_RADIUS,
    evaluation_budget,
)


# ---------------------------------------------------------------------------
# File utilities
# ---------------------------------------------------------------------------


def ensure_results_dir() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def write_dict_rows(
    path: Path,
    rows: list[dict],
) -> None:
    if not rows:
        return

    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------


def evaluate_final_controller(
    result: RunResult,
    evaluator: Evaluator,
) -> dict:
    distances = evaluator.evaluate_generalisation(
        result.best_genotype,
        N_TEST_POSITIONS,
        TEST_RADIUS,
    )

    return {
        "condition": result.condition,
        "seed": result.seed,
        "training_fitness": result.best_training_fitness,
        "test_mean": float(np.mean(distances)),
        "test_std": float(np.std(distances)),
        "test_median": float(np.median(distances)),
        "test_min": float(np.min(distances)),
        "test_max": float(np.max(distances)),
        **{
            f"test_position_{i:02d}": distance
            for i, distance in enumerate(distances)
        },
    }


def save_run(
    result: RunResult,
    final_row: dict,
) -> None:
    run_dir = (
        RESULTS_DIR
        / result.condition
        / f"seed_{result.seed}"
    )

    run_dir.mkdir(parents=True, exist_ok=True)

    write_dict_rows(
        run_dir / "history.csv",
        result.history,
    )

    if result.mutation_log:
        write_dict_rows(
            run_dir / "mutations.csv",
            result.mutation_log,
        )

    write_dict_rows(
        run_dir / "final.csv",
        [final_row],
    )

    np.save(
        run_dir / "best_genotype.npy",
        result.best_genotype,
    )


def run_one(
    condition: str,
    seed: int,
) -> dict:
    evaluator = Evaluator()

    if condition == "random":
        result = run_random_search(
            seed,
            evaluator,
            evaluation_budget(),
        )
    else:
        result = run_ea(
            condition,
            seed,
            evaluator,
        )

    final_row = evaluate_final_controller(
        result,
        evaluator,
    )

    save_run(
        result,
        final_row,
    )

    return final_row


# ---------------------------------------------------------------------------
# Load saved results
# ---------------------------------------------------------------------------


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as file:
        return list(csv.DictReader(file))


def load_all_final_results() -> list[dict]:
    rows = []

    for condition in (*CONDITIONS, "random"):
        for seed in FINAL_SEEDS:

            path = (
                RESULTS_DIR
                / condition
                / f"seed_{seed}"
                / "final.csv"
            )

            if not path.exists():
                continue

            raw = read_csv(path)[0]

            rows.append({
                "condition": condition,
                "seed": int(raw["seed"]),
                "training_fitness": float(raw["training_fitness"]),
                "test_mean": float(raw["test_mean"]),
                "test_std": float(raw["test_std"]),
            })

    return rows


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_learning_curves() -> None:
    plt.figure(figsize=(8, 5))

    for condition in (*CONDITIONS, "random"):

        histories = []

        for seed in FINAL_SEEDS:

            path = (
                RESULTS_DIR
                / condition
                / f"seed_{seed}"
                / "history.csv"
            )

            if not path.exists():
                continue

            rows = read_csv(path)

            evaluations = np.asarray(
                [float(row["evaluations"]) for row in rows]
            )

            best = np.asarray(
                [float(row["best_so_far"]) for row in rows]
            )

            histories.append(best)

        if not histories:
            continue

        values = np.asarray(histories)

        mean = np.mean(values, axis=0)
        std = np.std(values, axis=0)

        plt.plot(
            evaluations,
            mean,
            label=condition,
        )

        plt.fill_between(
            evaluations,
            mean - std,
            mean + std,
            alpha=0.2,
        )

    plt.xlabel("Fitness evaluations")
    plt.ylabel("Best-so-far distance to target (m)")
    plt.title("Evolutionary performance")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "learning_curves.png",
        dpi=300,
    )

    plt.close()


def plot_success_rate() -> None:
    plt.figure(figsize=(8, 5))

    for condition in CONDITIONS:

        histories = []

        for seed in FINAL_SEEDS:

            path = (
                RESULTS_DIR
                / condition
                / f"seed_{seed}"
                / "history.csv"
            )

            if not path.exists():
                continue

            rows = read_csv(path)[1:]

            generations = np.asarray(
                [int(float(row["generation"])) for row in rows]
            )

            success = np.asarray(
                [float(row["success_rate"]) for row in rows]
            )

            histories.append(success)

        if not histories:
            continue

        values = np.asarray(histories)

        mean = np.mean(values, axis=0)
        std = np.std(values, axis=0)

        plt.plot(
            generations,
            mean,
            label=condition,
        )

        plt.fill_between(
            generations,
            mean - std,
            mean + std,
            alpha=0.2,
        )

    plt.xlabel("Generation")
    plt.ylabel("Offspring success rate")
    plt.title("Mutation success rate")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "success_rate.png",
        dpi=300,
    )

    plt.close()


def plot_final_scores() -> None:
    rows = load_all_final_results()

    values = []

    labels = []

    for condition in (*CONDITIONS, "random"):

        condition_values = [
            row["test_mean"]
            for row in rows
            if row["condition"] == condition
        ]

        if condition_values:
            values.append(condition_values)
            labels.append(condition)

    plt.figure(figsize=(7, 5))

    plt.boxplot(
        values,
        tick_labels=labels,
    )

    plt.ylabel("Mean final distance over 20 positions (m)")
    plt.title("Final controller performance")
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "final_scores.png",
        dpi=300,
    )

    plt.close()


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------


def print_summary() -> None:
    rows = load_all_final_results()

    print("\nFINAL GENERALISATION RESULTS")
    print("=" * 60)

    for condition in (*CONDITIONS, "random"):

        scores = np.asarray([
            row["test_mean"]
            for row in rows
            if row["condition"] == condition
        ])

        if len(scores) == 0:
            continue

        print(
            f"{condition:12s} "
            f"n={len(scores):2d} | "
            f"mean={np.mean(scores):.4f} | "
            f"std={np.std(scores, ddof=1):.4f} | "
            f"median={np.median(scores):.4f}"
        )


# ---------------------------------------------------------------------------
# Inferential statistics
# ---------------------------------------------------------------------------


def statistical_tests() -> None:
    try:
        from scipy.stats import friedmanchisquare, wilcoxon
    except ImportError:
        print(
            "\nSciPy not installed: statistical tests skipped."
        )
        return

    rows = load_all_final_results()

    scores = {}

    for condition in CONDITIONS:

        condition_rows = {
            row["seed"]: row["test_mean"]
            for row in rows
            if row["condition"] == condition
        }

        scores[condition] = condition_rows

    common_seeds = sorted(
        set.intersection(
            *[
                set(scores[c].keys())
                for c in CONDITIONS
            ]
        )
    )

    if len(common_seeds) < 5:
        print(
            "\nNot enough complete paired runs "
            "for statistical testing."
        )
        return

    arrays = {
        condition: np.asarray([
            scores[condition][seed]
            for seed in common_seeds
        ])
        for condition in CONDITIONS
    }

    statistic, p_value = friedmanchisquare(
        arrays["sparse"],
        arrays["intermediate"],
        arrays["dense"],
    )

    print("\nFRIEDMAN TEST")
    print("=" * 60)
    print(f"chi-square = {statistic:.6f}")
    print(f"p          = {p_value:.6g}")

    # Pre-specified pairwise comparisons.
    comparisons = [
        ("sparse", "intermediate"),
        ("sparse", "dense"),
        ("intermediate", "dense"),
    ]

    tests = []

    for a, b in comparisons:

        statistic, p = wilcoxon(
            arrays[a],
            arrays[b],
            alternative="two-sided",
        )

        tests.append({
            "comparison": f"{a} vs {b}",
            "statistic": statistic,
            "p_raw": p,
        })

    # Holm correction.
    order = np.argsort(
        [test["p_raw"] for test in tests]
    )

    m = len(tests)

    adjusted = np.empty(m)

    running_max = 0.0

    for rank, index in enumerate(order):

        p_raw = tests[index]["p_raw"]

        p_adjusted = min(
            1.0,
            (m - rank) * p_raw,
        )

        running_max = max(
            running_max,
            p_adjusted,
        )

        adjusted[index] = running_max

    print("\nPAIRWISE WILCOXON + HOLM")
    print("=" * 60)

    for test, p_adjusted in zip(
        tests,
        adjusted,
        strict=True,
    ):
        print(
            f"{test['comparison']:28s} "
            f"W={test['statistic']:.3f} | "
            f"p={test['p_raw']:.6g} | "
            f"Holm={p_adjusted:.6g}"
        )


# ---------------------------------------------------------------------------
# Experimental design information
# ---------------------------------------------------------------------------


def print_design() -> None:
    evaluator = Evaluator()

    print("\nEXPERIMENTAL DESIGN")
    print("=" * 60)

    print(f"NN inputs          : {evaluator.input_size}")
    print(f"NN outputs         : {evaluator.output_size}")
    print(f"Total weights (n)  : {evaluator.num_weights}")
    print(f"Evaluation budget  : {evaluation_budget()}")

    print("\nMutation conditions")

    for condition in CONDITIONS:

        p, sigma = mutation_parameters(
            condition,
            evaluator.num_weights,
        )

        expected_mutations = evaluator.num_weights * p

        expected_squared_norm = (
            evaluator.num_weights
            * p
            * sigma**2
        )

        print(
            f"{condition:12s} "
            f"p={p:.6f} | "
            f"sigma={sigma:.6f} | "
            f"E[K]={expected_mutations:.2f} | "
            f"E[||dw||²]={expected_squared_norm:.6f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--condition",
        choices=(*CONDITIONS, "random"),
    )

    parser.add_argument(
        "--seed",
        type=int,
    )

    parser.add_argument(
        "--all",
        action="store_true",
    )

    parser.add_argument(
        "--analyse",
        action="store_true",
    )

    parser.add_argument(
        "--design",
        action="store_true",
    )

    args = parser.parse_args()

    ensure_results_dir()

    if args.design:
        print_design()
        return

    if args.all:

        for condition in (*CONDITIONS, "random"):

            for seed in FINAL_SEEDS:

                print("\n" + "=" * 60)
                print(
                    f"RUNNING {condition.upper()} "
                    f"| SEED {seed}"
                )
                print("=" * 60)

                run_one(
                    condition,
                    seed,
                )

        plot_learning_curves()
        plot_success_rate()
        plot_final_scores()

        print_summary()
        statistical_tests()

        return

    if args.analyse:
        plot_learning_curves()
        plot_success_rate()
        plot_final_scores()

        print_summary()
        statistical_tests()

        return

    if args.condition is None or args.seed is None:
        parser.error(
            "Use --condition CONDITION --seed SEED, "
            "--all, --analyse, or --design."
        )

    row = run_one(
        args.condition,
        args.seed,
    )

    print("\nFINAL RESULT")
    print(row)


if __name__ == "__main__":
    main()
