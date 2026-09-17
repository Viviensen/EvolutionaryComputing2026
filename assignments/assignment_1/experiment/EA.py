import copy
import random
import sys
from collections.abc import Callable
from pathlib import Path

import networkx as nx
import numpy as np

from ariel.ec import EA, EAOperation, Individual, Population
from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from ariel.ec.genotypes.tree.operators import random_tree, mutate_replace_node, mutate_subtree_replacement

from .settings import POP_SIZE, GENERATIONS, MAX_MODULES, MutationType


# exp1/ea_core.py -> assignment_1/A1_template_2026.py
sys.path.append(str(Path(__file__).resolve().parents[1]))

from A1_template_2026 import load_targets, fitness_function


type GenerationHistory = list[dict[str, float | int]]

TARGETS: list[nx.DiGraph] = load_targets()


def set_seed(seed: int) -> None:
    """Reset the random generators before each independent run."""
    random.seed(seed)
    np.random.seed(seed)


def create_individual() -> Individual:
    """Create one random tree-encoded robot."""
    individual = Individual()
    genome = random_tree(max_modules=MAX_MODULES)

    individual.genotype = genome.to_dict()
    individual.tags["parent"] = False

    return individual


def create_population() -> Population:
    """Create the initial population."""
    return Population([create_individual() for _ in range(POP_SIZE)])


def evaluate(population: Population) -> Population:
    """Evaluate all individuals that require evaluation."""

    for individual in population:
        if not individual.alive or not individual.requires_eval:
            continue

        genome = TreeGenome.from_dict(individual.genotype)
        body = genome.to_networkx()

        individual.fitness = fitness_function(body, TARGETS)
        individual.requires_eval = False

    return population


def parent_selection(population: Population) -> Population:
    """Select the best 50% of the population as parents."""

    population = population.sort(sort="min", attribute="fitness_")
    alive = [individual for individual in population if individual.alive]

    number_of_parents = len(alive) // 2

    for individual in alive:
        individual.tags["parent"] = False

    for individual in alive[:number_of_parents]:
        individual.tags["parent"] = True

    return population


def mutate(genome: TreeGenome, mutation_type: MutationType) -> TreeGenome:
    """Apply the mutation operator belonging to the EA variant."""

    child = copy.deepcopy(genome)

    if mutation_type == "replace":
        mutate_replace_node(child)

    elif mutation_type == "subtree":
        mutate_subtree_replacement(child, max_modules=MAX_MODULES)

    else:
        raise ValueError(f"Unknown mutation type: {mutation_type}")

    return child


def make_reproduction(mutation_type: MutationType) -> Callable[[Population], Population]:
    """Create POP_SIZE mutated offspring from selected parents."""

    def reproduction(population: Population) -> Population:
        parents = [individual for individual in population if individual.alive and individual.tags.get("parent", False)]

        if not parents:
            raise RuntimeError("No parents were selected.")

        offspring: list[Individual] = []

        while len(offspring) < POP_SIZE:
            parent = random.choice(parents)
            parent_genome = TreeGenome.from_dict(parent.genotype)
            child_genome = mutate(parent_genome, mutation_type)

            child = Individual()
            child.genotype = child_genome.to_dict()
            child.tags["parent"] = False

            offspring.append(child)

        population.extend(offspring)

        return population

    return reproduction


def record_generation(
    population: Population,
    generation: int,
    history: GenerationHistory,
) -> None:
    """Record fitness statistics of the surviving population."""

    alive = [individual for individual in population if individual.alive]
    fitnesses = np.array([individual.fitness_ for individual in alive])

    history.append({
        "generation": generation,
        "best": float(np.min(fitnesses)),
        "mean": float(np.mean(fitnesses)),
        "std": float(np.std(fitnesses)),
        "worst": float(np.max(fitnesses)),
    })


def make_survivor_selection(history: GenerationHistory) -> Callable[[Population], Population]:
    """Keep the POP_SIZE individuals with the lowest fitness."""

    generation_counter: dict[str, int] = {"generation": 0}

    def survivor_selection(population: Population) -> Population:
        population = population.sort(sort="min", attribute="fitness_")
        alive = [individual for individual in population if individual.alive]
        survivors = alive[:POP_SIZE]

        for individual in alive:
            if individual not in survivors:
                individual.alive = False

        generation_counter["generation"] += 1
        generation = generation_counter["generation"]

        record_generation(population, generation, history)

        fitnesses = [individual.fitness_ for individual in survivors]

        print(
            f"Generation {generation:3d} | "
            f"best = {min(fitnesses):.3f} | "
            f"mean = {np.mean(fitnesses):.3f} | "
            f"std = {np.std(fitnesses):.3f}"
        )

        return population

    return survivor_selection


def run_ea(mutation_type: MutationType, seed: int) -> tuple[Individual, GenerationHistory]:
    """Run one complete independent EA experiment."""

    print("\n" + "=" * 70)
    print(f"RUN | mutation = {mutation_type} | seed = {seed}")
    print("=" * 70)

    set_seed(seed)

    population = create_population()
    population = evaluate(population)

    history: GenerationHistory = []
    record_generation(population, generation=0, history=history)

    operations = [
        EAOperation(parent_selection),
        EAOperation(make_reproduction(mutation_type)),
        EAOperation(evaluate),
        EAOperation(make_survivor_selection(history)),
    ]

    ea = EA(population, operations=operations, num_steps=GENERATIONS, is_maximisation=False)
    ea.run()

    best = ea.get_solution("best", only_alive=True)

    return best, history


def run_random_search(seed: int) -> tuple[Individual, GenerationHistory]:
    """Run random search using the same number of evaluations as the EA."""

    set_seed(seed)

    number_of_evaluations: int = POP_SIZE + GENERATIONS * POP_SIZE
    best_fitness: float = np.inf
    best_individual: Individual | None = None
    history: GenerationHistory = []

    for evaluation in range(number_of_evaluations):
        individual = create_individual()
        evaluate(Population([individual]))

        if individual.fitness_ < best_fitness:
            best_fitness = individual.fitness_
            best_individual = individual

        if (evaluation + 1) % POP_SIZE == 0:
            generation = (evaluation + 1) // POP_SIZE - 1
            history.append({"generation": generation, "best": best_fitness})

    if best_individual is None:
        raise RuntimeError("Random search did not evaluate any individuals.")

    return best_individual, history
