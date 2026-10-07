"""Simple evolutionary algorithm for Assignment 2.

The experiment investigates one aspect only:
the strength (sigma) of Gaussian mutation.

The simulation itself follows A2_template_2026.py as closely as possible.
"""


import random
import sys
from pathlib import Path
from typing import cast

import mujoco as mj
import numpy as np
import numpy.typing as npt

from ariel.ec import EA, EAOperation, Individual, Population, set_seed
from ariel.utils.runners import simple_runner


# --------------------------------------------------------------------------- #
# IMPORT THE UNCHANGED TEMPLATE
# --------------------------------------------------------------------------- #

ASSIGNMENT_DIR = Path(__file__).resolve().parents[1]

if str(ASSIGNMENT_DIR) not in sys.path:
    sys.path.insert(0, str(ASSIGNMENT_DIR))

import A2_template_2026 as template


# --------------------------------------------------------------------------- #
# EA SETTINGS
# --------------------------------------------------------------------------- #

POP_SIZE = 20
GENERATIONS = 100
TOURNAMENT_SIZE = 2
INIT_STD = 0.5

type HistoryRow = dict[str, int | float]


# --------------------------------------------------------------------------- #
# GENOTYPE -> NEURAL NETWORK
# --------------------------------------------------------------------------- #

def genotype_to_weights(
    genotype: list[float],
    input_size: int,
    output_size: int,
) -> list[npt.NDArray[np.float64]]:
    """Convert a flat genotype into the two NN weight matrices."""

    genes = np.asarray(genotype, dtype=np.float64)

    split = input_size * template.HIDDEN_SIZE

    w1 = genes[:split].reshape(input_size, template.HIDDEN_SIZE)
    w2 = genes[split:].reshape(template.HIDDEN_SIZE, output_size)

    return [w1, w2]


def get_num_weights() -> int:
    """Determine the required genotype length."""

    mj.set_mjcb_control(None)

    world = template.build_world()
    robot = template.build_robot()

    world.spawn(
        robot.spec,
        position=template.SPAWN_POS,
        correct_collision_with_floor=True,
    )

    model = world.spec.compile()
    data = mj.MjData(model)

    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)

    input_size = len(data.qpos)
    output_size = int(model.nu)

    return (
        input_size * template.HIDDEN_SIZE
        + template.HIDDEN_SIZE * output_size
    )


# --------------------------------------------------------------------------- #
# RUN ONE INDIVIDUAL
# --------------------------------------------------------------------------- #

def evaluate_genotype(genotype: list[float]) -> float:
    """Evaluate one evolved controller.

    This follows run_experiment() from the template, except that the neural
    network weights come from the genotype instead of make_random_weights().
    """

    mj.set_mjcb_control(None)

    # Same world and robot as the template
    world = template.build_world()
    robot = template.build_robot()

    world.spawn(
        robot.spec,
        position=template.SPAWN_POS,
        correct_collision_with_floor=True,
    )

    # Same MuJoCo setup as the template
    model = world.spec.compile()
    data = mj.MjData(model)

    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)

    # Same network dimensions as the template
    input_size = len(data.qpos)
    output_size = int(model.nu)

    # The only difference:
    # use evolved weights instead of random weights
    weights = genotype_to_weights(
        genotype,
        input_size,
        output_size,
    )

    def control_callback(
        m: mj.MjModel,
        d: mj.MjData,
    ) -> None:
        actions = template.nn_controller(m, d, weights)
        d.ctrl[:] = actions

    initial_position = template.get_core_position(data)

    try:
        mj.set_mjcb_control(control_callback)

        simple_runner(
            model,
            data,
            duration=template.SIM_DURATION,
        )

    finally:
        mj.set_mjcb_control(None)

    final_position = template.get_core_position(data)

    return template.fitness_function(
        initial_position,
        final_position,
    )


# --------------------------------------------------------------------------- #
# INITIAL POPULATION
# --------------------------------------------------------------------------- #

def make_individual(
    rng: np.random.Generator,
    num_weights: int,
) -> Individual:
    """Create one random neural-network genotype."""

    individual = Individual()

    individual.genotype = rng.normal(
        loc=0.0,
        scale=INIT_STD,
        size=num_weights,
    ).tolist()

    return individual


# --------------------------------------------------------------------------- #
# EA OPERATIONS
# --------------------------------------------------------------------------- #

def evaluate_population(
    population: Population,
) -> Population:
    """Evaluate all individuals that still require evaluation."""

    for individual in population.unevaluated:

        genotype = cast(
            list[float],
            individual.genotype,
        )

        individual.fitness = evaluate_genotype(genotype)

    return population


def reproduce(
    population: Population,
    sigma: float,
    offspring_count: int,
    tournament_size: int,
    rng: np.random.Generator,
) -> Population:
    """Tournament selection followed by Gaussian mutation."""

    parents = population.alive.evaluated.to_list()

    for _ in range(offspring_count):

        competitors = [
            parents[int(rng.integers(len(parents)))]
            for _ in range(tournament_size)
        ]

        # Lower fitness = better
        parent = min(
            competitors,
            key=lambda individual: individual.fitness,
        )

        parent_genotype = np.asarray(
            cast(list[float], parent.genotype),
            dtype=np.float64,
        )

        mutation = rng.normal(
            loc=0.0,
            scale=sigma,
            size=len(parent_genotype),
        )

        child_genotype = parent_genotype + mutation

        child = Individual()
        child.genotype = child_genotype.tolist()

        population.append(child)

    return population


def survivor_selection(
    population: Population,
    population_size: int,
) -> Population:
    """Keep the best individuals after parents and offspring are combined."""

    ranked = population.alive.evaluated.sort(
        sort="min",
        attribute="fitness",
    )

    for individual in ranked[population_size:]:
        individual.alive = False

    return population


# --------------------------------------------------------------------------- #
# STATISTICS
# --------------------------------------------------------------------------- #

def population_stats(
    population: Population,
) -> dict[str, float]:
    """Return best and mean fitness of the living population."""

    fitness = np.asarray(
        [
            individual.fitness
            for individual in population.alive.evaluated
        ],
        dtype=np.float64,
    )

    return {
        "best": float(np.min(fitness)),
        "mean": float(np.mean(fitness)),
    }


# --------------------------------------------------------------------------- #
# RUN ONE EA
# --------------------------------------------------------------------------- #

def run_ea(
    seed: int,
    sigma: float,
    population_size: int = POP_SIZE,
    generations: int = GENERATIONS,
    tournament_size: int = TOURNAMENT_SIZE,
) -> tuple[list[HistoryRow], list[float]]:
    """Run the EA once for one seed and one mutation strength."""

    random.seed(seed)
    set_seed(seed)

    rng = np.random.default_rng(seed)

    num_weights = get_num_weights()

    initial_population = Population(
        [
            make_individual(rng, num_weights)
            for _ in range(population_size)
        ]
    )

    initial_population = evaluate_population(
        initial_population
    )

    history: list[HistoryRow] = [
        {
            "generation": 0,
            **population_stats(initial_population),
        }
    ]

    sigma_name = str(sigma).replace(".", "_")

    database = (
        Path("__data__")
        / "assignment_2"
        / f"sigma_{sigma_name}_seed_{seed}.db"
    )

    operations = [
        EAOperation(
            reproduce,
            sigma=sigma,
            offspring_count=population_size,
            tournament_size=tournament_size,
            rng=rng,
        ),
        EAOperation(
            evaluate_population,
        ),
        EAOperation(
            survivor_selection,
            population_size=population_size,
        ),
    ]

    ea = EA(
        initial_population,
        operations=operations,
        num_steps=generations,
        is_maximisation=False,
        quiet=True,
        db_file_path=database,
        db_handling="delete",
    )

    for generation in range(1, generations + 1):

        ea.step()

        # Reload population from database after ARIEL commits it
        ea.fetch_population(
            only_alive=True,
            requires_eval=False,
        )

        history.append(
            {
                "generation": generation,
                **population_stats(ea.population),
            }
        )

    best = ea.population.alive.evaluated.best(
        sort="min",
        attribute="fitness",
        n=1,
    )[0]

    best_genotype = [
        float(gene)
        for gene in cast(list[float], best.genotype)
    ]

    return history, best_genotype


# --------------------------------------------------------------------------- #
# RANDOM-SEARCH BASELINE
# --------------------------------------------------------------------------- #

def run_random_search(
    seed: int,
    population_size: int = POP_SIZE,
    generations: int = GENERATIONS,
) -> list[HistoryRow]:
    """Random-search baseline using the same evaluation budget as the EA."""

    rng = np.random.default_rng(seed)

    num_weights = get_num_weights()

    best_so_far = np.inf

    history: list[HistoryRow] = []

    for generation in range(generations + 1):

        generation_fitness: list[float] = []

        for _ in range(population_size):

            genotype = rng.normal(
                loc=0.0,
                scale=INIT_STD,
                size=num_weights,
            ).tolist()

            fitness = evaluate_genotype(genotype)

            generation_fitness.append(fitness)

            best_so_far = min(
                best_so_far,
                fitness,
            )

        history.append(
            {
                "generation": generation,
                "best": float(best_so_far),
                "mean": float(
                    np.mean(generation_fitness)
                ),
            }
        )

    return history