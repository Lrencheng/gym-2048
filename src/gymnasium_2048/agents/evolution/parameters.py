from __future__ import annotations

from dataclasses import asdict

import numpy as np

from gymnasium_2048.agents.evolution.config import PARAMETER_BOUNDS, PARAMETER_NAMES
from gymnasium_2048.agents.heuristic import HeuristicWeights


def clip_vector(vector: np.ndarray) -> np.ndarray:
    return np.clip(vector, PARAMETER_BOUNDS[:, 0], PARAMETER_BOUNDS[:, 1])


def vector_to_weights(vector: np.ndarray) -> HeuristicWeights:
    clipped = clip_vector(np.asarray(vector, dtype=np.float64))
    values = dict(zip(PARAMETER_NAMES, clipped.tolist()))
    return HeuristicWeights(**values)


def weights_to_vector(weights: HeuristicWeights) -> np.ndarray:
    data = asdict(weights)
    return np.array([data[name] for name in PARAMETER_NAMES], dtype=np.float64)


def weights_to_dict(weights: HeuristicWeights) -> dict[str, float]:
    return dict(zip(PARAMETER_NAMES, weights_to_vector(weights).tolist()))

