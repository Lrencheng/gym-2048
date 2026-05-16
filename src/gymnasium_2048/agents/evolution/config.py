from __future__ import annotations

from dataclasses import dataclass

import numpy as np


DEFAULT_SEED = 42

PARAMETER_NAMES = (
    "empty_cells",
    "smoothness",
    "monotonicity",
    "corner_max",
    "merge_potential",
    "edge_bonus",
    "reward",
)

PARAMETER_BOUNDS = np.array(
    [
        (5.0, 20.0),
        (0.5, 3.0),
        (1.0, 5.0),
        (2.0, 5.0),
        (2.0, 8.0),
        (0.01, 0.1),
        (0.5, 2.0),
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class EvolutionConfig:
    env_id: str = "gymnasium_2048:gymnasium_2048/TwentyFortyEight-v0"
    seed: int = DEFAULT_SEED
    population_size: int = 8
    generations: int = 5
    episodes_per_candidate: int = 5
    elite_size: int = 2
    tournament_size: int = 3
    crossover_rate: float = 0.8
    mutation_rate: float = 0.25
    mutation_scale: float = 0.15

    def validate(self) -> None:
        if self.population_size < 2:
            raise ValueError("population_size must be at least 2")
        if self.generations < 1:
            raise ValueError("generations must be at least 1")
        if self.episodes_per_candidate < 1:
            raise ValueError("episodes_per_candidate must be at least 1")
        if not 1 <= self.elite_size < self.population_size:
            raise ValueError("elite_size must be in [1, population_size)")
        if not 1 <= self.tournament_size <= self.population_size:
            raise ValueError("tournament_size must be in [1, population_size]")
        if not 0.0 <= self.crossover_rate <= 1.0:
            raise ValueError("crossover_rate must be in [0, 1]")
        if not 0.0 <= self.mutation_rate <= 1.0:
            raise ValueError("mutation_rate must be in [0, 1]")
        if self.mutation_scale < 0.0:
            raise ValueError("mutation_scale must be non-negative")

