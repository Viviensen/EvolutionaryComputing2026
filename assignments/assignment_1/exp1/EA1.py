import random
import copy
from pathlib import Path

import numpy as np

from ariel.ec import EA, EAOperation, Individual, Population

# load the tree genome representation and operators from ARIEL
from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from ariel.ec.genotypes.tree.operators import (
    random_tree,
    mutate_replace_node,
    mutate_subtree_replacement,
    mutate_shrink,
    mutate_hoist,
)

# Omdat EA1.py in exp1/ staat:
# exp1/EA1.py -> assignment_1/A1_template_2026.py
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from A1_template_2026 import (
    load_targets,
    fitness_function,
    show_body,
)


# ============================================================
# SETTINGS
# ============================================================

SEED = 42

POP_SIZE = 50
GENERATIONS = 5          # eerst testen; later bv. 100
MAX_MODULES = 20

random.seed(SEED)
RNG = np.random.default_rng(SEED)


# ============================================================
# INITIALIZATION
# ============================================================

def create_individual() -> Individual:
    """
    Create one random tree-encoded robot.
    """
    # create a new individual
    individual = Individual()

    # create genome and store it as a dictionary in the individual
    genome = random_tree(max_modules=MAX_MODULES)
    individual.genotype = genome.to_dict()

    # new individuals get the "parent" tag set to False
    individual.tags["parent"] = False
    return individual


def create_population() -> Population:
    """
    Create the initial population.
    """
    # create a list of random individuals
    individuals = [create_individual()for _ in range(POP_SIZE)]
    return Population(individuals)


TARGETS = load_targets()


def evaluate(population: Population) -> Population:
    """Evaluate every individual that requires evaluation."""

    for individual in population:

        # check whether individual is still evolvable
        if not individual.alive or not individual.requires_eval:
            continue

        # decode the genotype into a TreeGenome
        genome = TreeGenome.from_dict(individual.genotype)

        # decode the TreeGenome into a NetworkX DiGraph for evaluation
        body = genome.to_networkx()

        # compute the fitness w.r.t. target bodies
        fitness = fitness_function(body, TARGETS,)

        # assign the fitness and mark individual as evaluated
        individual.fitness = fitness
        individual.requires_eval = False

    return population


# ============================================================
# PARENT SELECTION
# ============================================================

def parent_selection(population: Population) -> Population:
    """
    Select the best 50% of the population as parents.
    Fitness is minimized.
    """

    population = population.sort(sort="min",attribute="fitness_",)

    alive = [ind for ind in population if ind.alive]

    number_of_parents = len(alive) // 2

    # reset tags
    for individual in alive:
        individual.tags["parent"] = False

    # best half becomes parent
    for individual in alive[:number_of_parents]:
        individual.tags["parent"] = True

    return population


# ============================================================
# MUTATION
# ============================================================

def mutate(genome: TreeGenome) -> TreeGenome:
    """
    Apply one random tree mutation.
    The four operators are provided by ARIEL.
    """

    child = copy.deepcopy(genome)

    mutation_type = RNG.choice(
        [
            "replace",
            "subtree",
            "shrink",
            "hoist",
        ]
    )

    if mutation_type == "replace":
        mutate_replace_node(child)

    elif mutation_type == "subtree":
        mutate_subtree_replacement(
            child,
            max_modules=MAX_MODULES,
        )

    elif mutation_type == "shrink":
        mutate_shrink(child)

    elif mutation_type == "hoist":
        mutate_hoist(child)

    return child


# ============================================================
# REPRODUCTION
# ============================================================

def reproduction(population: Population) -> Population:
    """
    Create offspring from selected parents.
    EA1 is mutation-only:
    parent -> copy -> mutation -> child
    """

    parents = [
        ind
        for ind in population
        if ind.alive
        and ind.tags.get("parent", False)
    ]

    offspring = []

    # Create POP_SIZE children
    while len(offspring) < POP_SIZE:

        parent = random.choice(parents)

        parent_genome = TreeGenome.from_dict(
            parent.genotype
        )

        child_genome = mutate(
            parent_genome
        )

        child = Individual()

        child.genotype = child_genome.to_dict()
        child.tags["parent"] = False

        offspring.append(child)

    population.extend(offspring)

    return population


# ============================================================
# SURVIVOR SELECTION
# ============================================================

def survivor_selection(population: Population) -> Population:
    """
    Keep the POP_SIZE individuals with lowest fitness.
    Parents and offspring compete together.
    """

    population = population.sort(
        sort="min",
        attribute="fitness_",
    )

    alive = [
        ind for ind in population
        if ind.alive
    ]

    survivors = alive[:POP_SIZE]

    for individual in alive:

        if individual not in survivors:
            individual.alive = False

    # useful while debugging
    fitnesses = [
        ind.fitness_
        for ind in survivors
    ]

    print(
        f"best = {min(fitnesses):.3f} | "
        f"mean = {np.mean(fitnesses):.3f} | "
        f"worst = {max(fitnesses):.3f}"
    )

    return population


# ============================================================
# RUN EA
# ============================================================

def run_ea():

    print("Creating population...")

    population = create_population()

    # Generation 0 needs fitness before selection
    population = evaluate(population)

    operations = [
        EAOperation(parent_selection),
        EAOperation(reproduction),
        EAOperation(evaluate),
        EAOperation(survivor_selection),
    ]

    ea = EA(
        population,
        operations=operations,
        num_steps=GENERATIONS,
        is_maximisation=False,
    )

    ea.run()

    best = ea.get_solution(
        "best",
        only_alive=True,
    )

    return best


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    best = run_ea()

    print("\nFINAL RESULT")
    print("fitness:", best.fitness_)

    genome = TreeGenome.from_dict(
        best.genotype
    )

    body = genome.to_networkx()

    print(
        "modules:",
        body.number_of_nodes(),
    )

    show_body(
        body,
        mode="frame",
        file_name="EA1_best",
    )