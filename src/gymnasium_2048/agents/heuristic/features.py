from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np

RewardTransform = Literal["raw", "log2p1", "none"]


@dataclass(frozen=True)
class HeuristicWeights:
    empty_cells: float = 12.271004
    smoothness: float = 0.859501
    monotonicity: float = 2.891370
    corner_max: float = 3.555273
    merge_potential: float = 2.732829
    edge_bonus: float = 0.098447
    reward: float = 1.021244
"""
进化后参数
    empty_cells: 12.271004
    smoothness: 0.859501
    monotonicity: 2.891370
    corner_max: 3.555273
    merge_potential: 2.732829
    edge_bonus: 0.098447
    reward: 1.021244
baseline参数:
    empty_cells: float = 12.0
    smoothness: float = 1.0
    monotonicity: float = 2.0
    corner_max: float = 4.0
    merge_potential: float = 6.0
    edge_bonus: float = 0.02
    reward: float = 1.0
"""


DEFAULT_WEIGHTS = HeuristicWeights()

SNAKE_WEIGHTS = np.array(
    [
        [15.0, 14.0, 13.0, 12.0],
        [8.0, 9.0, 10.0, 11.0],
        [7.0, 6.0, 5.0, 4.0],
        [0.0, 1.0, 2.0, 3.0],
    ],
    dtype=np.float64,
)


def transform_reward(reward: float, transform: RewardTransform = "raw") -> float:
    if transform == "raw":
        return float(reward)
    if transform == "log2p1":
        return float(np.log2(reward + 1.0)) if reward > 0 else 0.0
    if transform == "none":
        return 0.0
    raise ValueError(f"unknown reward_transform: {transform!r}")


def actual_tile_values(board: np.ndarray) -> np.ndarray:
    values = np.zeros_like(board, dtype=np.int64)
    non_empty = board > 0
    values[non_empty] = np.power(2, board[non_empty], dtype=np.int64)
    return values


def empty_cells(board: np.ndarray) -> float:
    return float(np.count_nonzero(board == 0))


def smoothness(board: np.ndarray) -> float:
    score = 0.0
    rows, cols = board.shape

    for row in range(rows):
        for col in range(cols):
            value = int(board[row, col])
            if value == 0:
                continue

            if col + 1 < cols and board[row, col + 1] != 0:
                score -= abs(value - int(board[row, col + 1]))

            if row + 1 < rows and board[row + 1, col] != 0:
                score -= abs(value - int(board[row + 1, col]))

    return score


@lru_cache(maxsize=200_000)
def _line_monotonicity_cached(line: tuple[int, ...]) -> float:
    non_empty = [value for value in line if value != 0]
    if len(non_empty) < 2:
        return 0.0

    increasing_score = 0.0
    decreasing_score = 0.0
    for current, next_value in zip(non_empty, non_empty[1:]):
        diff = next_value - current
        increasing_score -= max(0, -diff)
        decreasing_score -= max(0, diff)

    return max(increasing_score, decreasing_score)


def _line_monotonicity(line: np.ndarray) -> float:
    return _line_monotonicity_cached(tuple(int(value) for value in line))


def monotonicity(board: np.ndarray) -> float:
    score = 0.0

    for row in range(board.shape[0]):
        score += _line_monotonicity(board[row, :])

    for col in range(board.shape[1]):
        score += _line_monotonicity(board[:, col])

    return score


def corner_max(board: np.ndarray) -> float:
    max_value = int(np.max(board))
    if max_value == 0:
        return 0.0

    corners = (
        board[0, 0],
        board[0, -1],
        board[-1, 0],
        board[-1, -1],
    )
    return float(max_value) if any(corner == max_value for corner in corners) else 0.0


def merge_potential(board: np.ndarray) -> float:
    score = 0.0
    rows, cols = board.shape

    for row in range(rows):
        for col in range(cols):
            value = board[row, col]
            if value == 0:
                continue

            if col + 1 < cols and value == board[row, col + 1]:
                score += 1.0

            if row + 1 < rows and value == board[row + 1, col]:
                score += 1.0

    return score


def edge_bonus(board: np.ndarray) -> float:
    values = actual_tile_values(board)
    if values.size == 0:
        return 0.0

    edge_mask = np.zeros(board.shape, dtype=bool)
    edge_mask[0, :] = True
    edge_mask[-1, :] = True
    edge_mask[:, 0] = True
    edge_mask[:, -1] = True

    return float(np.sum(values[edge_mask]))


def max_tile(board: np.ndarray) -> float:
    return float(np.max(board))


def snake_score(board: np.ndarray) -> float:
    return float(np.sum(np.asarray(board, dtype=np.float64) * SNAKE_WEIGHTS) / 16.0)


def evaluate_board(
    board: np.ndarray,
    reward: float = 0.0,
    weights: HeuristicWeights = DEFAULT_WEIGHTS,
    reward_transform: RewardTransform = "raw",
) -> float:
    reward_score = transform_reward(reward, reward_transform)
    return (
        weights.empty_cells * empty_cells(board)
        + weights.smoothness * smoothness(board)
        + weights.monotonicity * monotonicity(board)
        + weights.corner_max * corner_max(board)
        + weights.merge_potential * merge_potential(board)
        + weights.edge_bonus * edge_bonus(board)
        + weights.reward * reward_score
    )
