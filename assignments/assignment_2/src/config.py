"""Configuration for Assignment 2."""

from pathlib import Path

# Paths
ASSIGNMENT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ASSIGNMENT_DIR / "results"

# Simulation
SPAWN_POS = (0.0, 0.0, 0.1)
TARGET_POS = (2.0, 0.0, 0.1)
SIM_DURATION = 15.0

# Neural network
HIDDEN_SIZE = 8
RHYTHM_FREQUENCY = 1.0

# EA
MU = 30
LAMBDA = 30
GENERATIONS = 100
TOURNAMENT_SIZE = 2
INITIAL_WEIGHT_STD = 0.5

# Mutation experiment
BASE_MUTATION_PROBABILITY = 0.10
BASE_SIGMA = 0.20

CONDITIONS = (
    "sparse",
    "intermediate",
    "dense",
)

# Independent runs
FINAL_SEEDS = tuple(range(10))

# Generalisation test
N_TEST_POSITIONS = 20
TEST_RADIUS = 2.0


def evaluation_budget() -> int:
    """Number of fitness evaluations in one EA run."""
    return MU + GENERATIONS * LAMBDA
