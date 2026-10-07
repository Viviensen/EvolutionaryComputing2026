"""Evolutionary algorithm for Assignment 2."""

from typing import Any

import numpy as np
import numpy.typing as npt

from ariel.ec import Individual, Population, set_seed
from config import BASE_MUTATION_PROBABILITY, BASE_SIGMA, GENERATIONS, INITIAL_WEIGHT_STD, LAMBDA, MU, TOURNAMENT_SIZE
from eval import Evaluator

Array = npt.NDArray[np.float64]


def mutation_parameters(condition: str, n_weights: int) -> tuple[float, float]:
    """Return mutation probability and sigma."""

    if condition == "sparse":
        p = 1.0 / n_weights
    elif condition == "intermediate":
        p = BASE_MUTATION_PROBABILITY
    elif condition == "dense":
        p = 1.0
    else:
        raise ValueError("Unknown mutation condition")

    sigma = BASE_SIGMA * np.sqrt(BASE_MUTATION_PROBABILITY / p)
    return float(p), float(sigma)


def mutate(genotype: Array, condition: str, rng: np.random.Generator) -> tuple[Array, int]:
    """Apply Gaussian mutation."""

    p, sigma = mutation_parameters(condition, len(genotype))
    mask = rng.random(len(genotype)) < p
    noise = rng.normal(0.0, sigma, len(genotype))

    child = genotype + mask * noise
    changed_weights = int(np.sum(mask))

    return child, changed_weights


def make_individual(genotype: Array, fitness: float) -> Individual:
    individual = Individual()
    individual.genotype = genotype.tolist()
    individual.fitness = fitness
    return individual


def get_genotype(individual: Individual) -> Array:
    return np.asarray(individual.genotype, dtype=np.float64)


def get_fitness(individual: Individual) -> float:
    return float(individual.fitness)


def initialize_population(evaluator: Evaluator, seed: int) -> Population:
    """Create initial population."""

    rng = np.random.default_rng(seed)
    individuals: list[Individual] = []

    for _ in range(MU):
        genotype = rng.normal(0.0, INITIAL_WEIGHT_STD, evaluator.num_weights)
        fitness = evaluator.evaluate(genotype)
        individuals.append(make_individual(genotype, fitness))

    return Population(individuals)


def tournament_selection(population: Population, rng: np.random.Generator) -> Individual:
    """Select one parent."""

    indices = rng.choice(len(population), TOURNAMENT_SIZE, replace=False)
    candidates = [population[int(i)] for i in indices]

    return min(candidates, key=get_fitness)


def run_ea(condition: str, seed: int) -> dict[str, Any]:
    """Run one evolutionary experiment."""

    set_seed(seed)

    evaluator = Evaluator()
    selection_rng = np.random.default_rng(seed + 1000)
    mutation_rng = np.random.default_rng(seed + 2000)
    population = initialize_population(evaluator, seed)

    history: list[dict[str, float]] = []

    for generation in range(GENERATIONS):
        offspring: list[Individual] = []
        successes = 0
        changed_weights: list[int] = []
        improvements: list[float] = []

        for _ in range(LAMBDA):
            # Define parent
            parent = tournament_selection(population, selection_rng)
            parent_genotype = get_genotype(parent)
            parent_fitness = get_fitness(parent)

            # Define child
            child_genotype, n_changed = mutate(parent_genotype, condition, mutation_rng)
            child_fitness = evaluator.evaluate(child_genotype)

            offspring.append(make_individual(child_genotype, child_fitness))
            changed_weights.append(n_changed)

            if child_fitness < parent_fitness:
                successes += 1
                improvements.append(parent_fitness - child_fitness)

        # (mu + lambda) survivor selection
        combined = list(population) + offspring
        combined.sort(key=get_fitness)
        population = Population(combined[:MU])

        fitness_values = np.asarray([get_fitness(individual) for individual in population], dtype=np.float64)
        mean_improvement = float(np.mean(improvements)) if improvements else 0.0

        history.append({
            "generation": float(generation + 1),
            "best": float(np.min(fitness_values)),
            "mean": float(np.mean(fitness_values)),
            "std": float(np.std(fitness_values)),
            "success_rate": float(successes / LAMBDA),
            "mean_changed_weights": float(np.mean(changed_weights)),
            "mean_improvement": mean_improvement
        })

        print(
            f"{condition:12s} | "
            f"seed {seed} | "
            f"generation {generation + 1:3d} | "
            f"best {np.min(fitness_values):.4f}"
        )

    best = min(population, key=get_fitness)

    return {
        "condition": condition,
        "seed": seed,
        "best_genotype": get_genotype(best),
        "best_fitness": get_fitness(best),
        "history": history
    }


def run_random_search(seed: int) -> dict[str, Any]:
    """Random-search baseline with equal evaluation budget."""

    evaluator = Evaluator()
    rng = np.random.default_rng(seed)

    budget = MU + GENERATIONS * LAMBDA
    best_genotype: Array | None = None
    best_fitness = np.inf
    history: list[dict[str, float]] = []
    fitness_values: list[float] = []

    for evaluation in range(1, budget + 1):
        genotype = rng.normal(0.0, INITIAL_WEIGHT_STD, evaluator.num_weights)
        fitness = evaluator.evaluate(genotype)
        fitness_values.append(fitness)

        if fitness < best_fitness:
            best_fitness = fitness
            best_genotype = genotype.copy()

        if evaluation >= MU and (evaluation - MU) % LAMBDA == 0:
            generation = (evaluation - MU) // LAMBDA

            history.append({
                "generation": float(generation),
                "best": float(best_fitness),
                "mean": float(np.mean(fitness_values)),
                "std": float(np.std(fitness_values))
            })

    if best_genotype is None:
        raise RuntimeError("Random search produced no genotype.")

    return {
        "condition": "random",
        "seed": seed,
        "best_genotype": best_genotype,
        "best_fitness": float(best_fitness),
        "history": history
    }