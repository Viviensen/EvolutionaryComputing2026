import csv
import statistics as stats

import matplotlib.pyplot as plt
import numpy as np

from ariel.ec.genotypes.tree.tree_genome import TreeGenome

from tree_edit_distance import distances_to_targets

from A1_template_2026 import show_body
from .EA import run_ea, run_random_search, TARGETS
from .settings import SEEDS, VARIANTS, RESULTS_DIR, CAM_FOVY


type HistoryRow = dict[str, float | int]
type History = list[HistoryRow]
type AllHistories = dict[str, list[History]]
type FinalResultRow = dict[str, float | int | str]
type PerTargetRow = dict[str, float | int | str]


# ============================================================
# SAVE RESULTS
# ============================================================

def save_history(history: History, variant: str, seed: int) -> None:
    """Save generation statistics for one run."""

    file_name = RESULTS_DIR / f"history_{variant}_seed_{seed}.csv"

    with open(file_name, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def save_final_results(results: list[FinalResultRow]) -> None:
    """Save the final fitness and morphology size of every run."""

    file_name = RESULTS_DIR / "final_results.csv"

    with open(file_name, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["variant", "seed", "best_fitness", "modules"])
        writer.writeheader()
        writer.writerows(results)


def summarize_final_results() -> None:
    """Aggregate final_results.csv into mean +/- std per variant."""

    with open(RESULTS_DIR / "final_results.csv", newline="") as file:
        rows = list(csv.DictReader(file))

    by_variant: dict[str, list[dict[str, float]]] = {}
    for row in rows:
        by_variant.setdefault(row["variant"], []).append({
            "fitness": float(row["best_fitness"]),
            "modules": float(row["modules"]),
        })

    summary_path = RESULTS_DIR / "summary_table.csv"
    with open(summary_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "variant", "n", "fitness_mean", "fitness_std", "modules_mean", "modules_std",
        ])
        for variant, entries in by_variant.items():
            fitness = [e["fitness"] for e in entries]
            modules = [e["modules"] for e in entries]
            writer.writerow([
                variant,
                len(entries),
                round(stats.mean(fitness), 3),
                round(stats.pstdev(fitness), 3),
                round(stats.mean(modules), 3),
                round(stats.pstdev(modules), 3),
            ])

    print(f"saved {summary_path}")


def save_per_target_distances(rows: list[PerTargetRow]) -> None:
    """Save each best individual's tree-edit distance to every target separately."""

    file_name = RESULTS_DIR / "per_target_distances.csv"
    fieldnames = ["variant", "seed"] + [f"target_{i:02d}" for i in range(len(TARGETS))]

    with open(file_name, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved {file_name}")


# ============================================================
# PLOTS
# ============================================================

def plot_best_fitness(all_histories: AllHistories) -> None:
    """Plot average best fitness +/- standard deviation over the runs."""

    plt.figure()

    for variant, histories in all_histories.items():
        values = np.array([[generation["best"] for generation in history] for history in histories])

        generations = np.arange(values.shape[1])
        mean = np.mean(values, axis=0)
        std = np.std(values, axis=0)

        plt.plot(generations, mean, label=variant)
        plt.fill_between(generations, mean - std, mean + std, alpha=0.2)

    plt.xlabel("Generation")
    plt.ylabel("Best fitness")
    plt.title("Best fitness over generations")
    plt.legend()
    plt.tight_layout()

    plt.savefig(RESULTS_DIR / "best_fitness_convergence.png", dpi=300)
    plt.close()


def plot_mean_fitness(all_histories: AllHistories) -> None:
    """Plot average population fitness for EA1 and EA2."""

    plt.figure()

    for variant in VARIANTS:
        histories = all_histories[variant]
        values = np.array([[generation["mean"] for generation in history] for history in histories])

        generations = np.arange(values.shape[1])
        mean = np.mean(values, axis=0)
        std = np.std(values, axis=0)

        plt.plot(generations, mean, label=variant)
        plt.fill_between(generations, mean - std, mean + std, alpha=0.2)

    plt.xlabel("Generation")
    plt.ylabel("Mean population fitness")
    plt.title("Mean population fitness over generations")
    plt.legend()
    plt.tight_layout()

    plt.savefig(RESULTS_DIR / "mean_fitness_convergence.png", dpi=300)
    plt.close()


# ============================================================
# RUN EXPERIMENT
# ============================================================

def run_experiment() -> tuple[AllHistories, list[FinalResultRow]]:

    all_histories: AllHistories = {variant: [] for variant in VARIANTS}
    all_histories["random_search"] = []

    final_results: list[FinalResultRow] = []
    per_target_results: list[PerTargetRow] = []

    # EA1 and EA2
    for variant, mutation_type in VARIANTS.items():
        for seed in SEEDS:
            best, history = run_ea(mutation_type, seed)

            save_history(history, variant, seed)
            all_histories[variant].append(history)

            genome = TreeGenome.from_dict(best.genotype)
            body = genome.to_networkx()

            final_results.append({
                "variant": variant,
                "seed": seed,
                "best_fitness": best.fitness_,
                "modules": body.number_of_nodes(),
            })

            per_target_results.append({
                "variant": variant,
                "seed": seed,
                **{
                    f"target_{i:02d}": distance
                    for i, distance in enumerate(distances_to_targets(body, TARGETS))
                },
            })

            show_body(body, mode="frame", file_name=str(RESULTS_DIR / f"best_{variant}_seed_{seed}"), cam_fovy=CAM_FOVY)

            print(
                f"\nFINAL | {variant} | seed = {seed} | "
                f"fitness = {best.fitness_:.3f} | modules = {body.number_of_nodes()}"
            )

    # Random-search baseline
    for seed in SEEDS:
        best, history = run_random_search(seed)

        save_history(history, "random_search", seed)
        all_histories["random_search"].append(history)

        genome = TreeGenome.from_dict(best.genotype)
        body = genome.to_networkx()

        final_results.append({
            "variant": "random_search",
            "seed": seed,
            "best_fitness": best.fitness_,
            "modules": body.number_of_nodes(),
        })

        per_target_results.append({
            "variant": "random_search",
            "seed": seed,
            **{
                f"target_{i:02d}": distance
                for i, distance in enumerate(distances_to_targets(body, TARGETS))
            },
        })

    save_final_results(final_results)
    summarize_final_results()
    save_per_target_distances(per_target_results)

    plot_best_fitness(all_histories)
    plot_mean_fitness(all_histories)

    return all_histories, final_results


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    histories, results = run_experiment()

    print("\n" + "=" * 70)
    print("EXPERIMENT FINISHED")
    print("=" * 70)

    for result in results:
        print(
            f"{result['variant']:25s} | "
            f"seed = {result['seed']} | "
            f"fitness = {result['best_fitness']:.3f} | "
            f"modules = {result['modules']}"
        )
    summarize_final_results()
