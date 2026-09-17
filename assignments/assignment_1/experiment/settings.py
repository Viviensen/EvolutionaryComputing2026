# This file contains the settings for the evolutionary computing experiment.
from pathlib import Path
from typing import Literal

type MutationType = Literal["replace", "subtree"]

POP_SIZE: int = 50
GENERATIONS: int = 100
MAX_MODULES: int = 20

SEEDS: list[int] = [42, 43, 44, 45, 46]

VARIANTS: dict[str, MutationType] = {
    "replace_node": "replace",
    "subtree_replacement": "subtree",
}

RESULTS_DIR: Path = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
