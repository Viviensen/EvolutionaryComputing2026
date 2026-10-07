# This file contains the settings for the neuroevolution experiment (Assignment 2).
from pathlib import Path
from typing import Literal

import numpy as np

from A2_template_2026 import TARGET_POSITION

type Condition = Literal["sparse", "intermediate", "dense"]

# --- EA ---
POP_SIZE: int = 30          # mu: parents kept each generation
OFFSPRING_SIZE: int = 30    # lambda: children made each generation
GENERATIONS: int = 100      # after the initial population
TOURNAMENT_SIZE: int = 2
INIT_STD: float = 0.5       # initial weights ~ N(0, 0.5^2)

# Simulations per run (random search gets the same budget)
BUDGET: int = POP_SIZE + GENERATIONS * OFFSPRING_SIZE

# --- Seeds ---
SEEDS: list[int] = list(range(10))        # final runs
PILOT_SEEDS: list[int] = [100, 101, 102]  # pilots only

# --- Mutation conditions ---
# dw_i = B_i * eps_i with B_i ~ Bernoulli(p), eps_i ~ N(0, sigma^2); n * p * sigma^2 is constant
REFERENCE_P: float = 0.10
REFERENCE_SIGMA: float = 0.20
CONDITIONS: list[Condition] = ["sparse", "intermediate", "dense"]


def mutation_parameters(condition: Condition, n: int) -> tuple[float, float]:
    """Return (p, sigma) for a condition, given n = number of weights."""
    p = {"sparse": 1 / n, "intermediate": 0.10, "dense": 1.0}[condition]
    sigma = REFERENCE_SIGMA * np.sqrt(REFERENCE_P / p)
    return p, sigma


# --- Test positions: 20 points on a circle of radius 2 m around the target ---
TEST_RADIUS: float = 2.0
NUM_TEST_POSITIONS: int = 20
TEST_POSITIONS: list[list[float]] = [
    [
        TARGET_POSITION[0] + TEST_RADIUS * np.cos(2 * np.pi * i / NUM_TEST_POSITIONS),
        TARGET_POSITION[1] + TEST_RADIUS * np.sin(2 * np.pi * i / NUM_TEST_POSITIONS),
        TARGET_POSITION[2],
    ]
    for i in range(NUM_TEST_POSITIONS)
]

# --- Output ---
RESULTS_DIR: Path = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
