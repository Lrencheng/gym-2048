from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from gymnasium_2048.agents.heuristic.features import (
    RewardTransform,
    corner_max,
    empty_cells,
    max_tile,
    merge_potential,
    monotonicity,
    smoothness,
    snake_score,
    transform_reward,
)


@dataclass(frozen=True)
class ExpectimaxHeuristicWeights:
    empty_cells: float = 3.0
    smoothness: float = 0.35
    monotonicity: float = 1.2
    merge_potential: float = 1.0
    corner_max: float = 1.4
    max_tile: float = 0.6
    snake: float = 0.8
    reward: float = 1.0


DEFAULT_WEIGHTS = ExpectimaxHeuristicWeights()


def _weights_key(weights: ExpectimaxHeuristicWeights) -> tuple[float, ...]:
    return (
        weights.empty_cells,
        weights.smoothness,
        weights.monotonicity,
        weights.merge_potential,
        weights.corner_max,
        weights.max_tile,
        weights.snake,
        weights.reward,
    )


@lru_cache(maxsize=200_000)
def _evaluate_board_cached(
    board_bytes: bytes,
    shape: tuple[int, int],
    reward: float,
    reward_transform: RewardTransform,
    weights_key: tuple[float, ...],
) -> float:
    board = np.frombuffer(board_bytes, dtype=np.uint8).reshape(shape)
    weights = ExpectimaxHeuristicWeights(*weights_key)
    state = board.astype(np.float64, copy=False)
    reward_score = transform_reward(reward, reward_transform)

    return float(
        weights.empty_cells * empty_cells(state)
        + weights.smoothness * smoothness(state)
        + weights.monotonicity * monotonicity(state)
        + weights.merge_potential * merge_potential(state)
        + weights.corner_max * corner_max(state)
        + weights.max_tile * max_tile(state)
        + weights.snake * snake_score(state)
        + weights.reward * reward_score
    )


def evaluate_board(
    board: np.ndarray,
    reward: float = 0.0,
    weights: ExpectimaxHeuristicWeights = DEFAULT_WEIGHTS,
    reward_transform: RewardTransform = "log2p1",
) -> float:
    """Evaluate an exponent board without allowing raw tile values to explode."""
    state = np.ascontiguousarray(board, dtype=np.uint8)
    return _evaluate_board_cached(
        state.tobytes(),
        state.shape,
        float(reward),
        reward_transform,
        _weights_key(weights),
    )
