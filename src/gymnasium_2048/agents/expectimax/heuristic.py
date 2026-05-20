from __future__ import annotations

from functools import lru_cache
from dataclasses import dataclass

import numpy as np

from gymnasium_2048.agents.expectimax.board import count_empty


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

SNAKE_WEIGHTS = np.array(
    [
        [15.0, 14.0, 13.0, 12.0],
        [8.0, 9.0, 10.0, 11.0],
        [7.0, 6.0, 5.0, 4.0],
        [0.0, 1.0, 2.0, 3.0],
    ],
    dtype=np.float64,
)


def smoothness(board: np.ndarray) -> float:
    state = np.asarray(board, dtype=np.int16)
    horizontal_mask = (state[:, :-1] != 0) & (state[:, 1:] != 0)
    vertical_mask = (state[:-1, :] != 0) & (state[1:, :] != 0)
    horizontal = np.abs(state[:, :-1] - state[:, 1:])[horizontal_mask].sum()
    vertical = np.abs(state[:-1, :] - state[1:, :])[vertical_mask].sum()
    return -float(horizontal + vertical)


@lru_cache(maxsize=200_000)
def _line_monotonicity_cached(line: tuple[int, ...]) -> float:
    non_empty = [value for value in line if value != 0]
    if len(non_empty) < 2:
        return 0.0

    increasing = 0.0
    decreasing = 0.0
    for current, next_value in zip(non_empty, non_empty[1:]):
        diff = next_value - current
        increasing -= max(0, -diff)
        decreasing -= max(0, diff)
    return max(increasing, decreasing)


def _line_monotonicity(line: np.ndarray) -> float:
    return _line_monotonicity_cached(tuple(int(value) for value in line))


def monotonicity(board: np.ndarray) -> float:
    state = np.asarray(board)
    return float(
        sum(_line_monotonicity(state[row, :]) for row in range(state.shape[0]))
        + sum(_line_monotonicity(state[:, col]) for col in range(state.shape[1]))
    )


def merge_potential(board: np.ndarray) -> float:
    state = np.asarray(board)
    horizontal = (state[:, :-1] != 0) & (state[:, :-1] == state[:, 1:])
    vertical = (state[:-1, :] != 0) & (state[:-1, :] == state[1:, :])
    return float(np.count_nonzero(horizontal) + np.count_nonzero(vertical))


def corner_max(board: np.ndarray) -> float:
    max_value = int(np.max(board))
    if max_value == 0:
        return 0.0
    corners = (board[0, 0], board[0, -1], board[-1, 0], board[-1, -1])
    return float(max_value) if any(int(corner) == max_value for corner in corners) else 0.0


def snake_score(board: np.ndarray) -> float:
    """Reward large exponent tiles following a snake-like ordering."""
    return float(np.sum(np.asarray(board, dtype=np.float64) * SNAKE_WEIGHTS) / 16.0)


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
    weights_key: tuple[float, ...],
) -> float:
    board = np.frombuffer(board_bytes, dtype=np.uint8).reshape(shape)
    weights = ExpectimaxHeuristicWeights(*weights_key)
    state = board.astype(np.float64, copy=False)
    reward_score = np.log2(reward + 1.0) if reward > 0 else 0.0
    max_exponent = float(np.max(state))

    return float(
        weights.empty_cells * count_empty(state)
        + weights.smoothness * smoothness(state)
        + weights.monotonicity * monotonicity(state)
        + weights.merge_potential * merge_potential(state)
        + weights.corner_max * corner_max(state)
        + weights.max_tile * max_exponent
        + weights.snake * snake_score(state)
        + weights.reward * reward_score
    )


def evaluate_board(
    board: np.ndarray,
    reward: float = 0.0,
    weights: ExpectimaxHeuristicWeights = DEFAULT_WEIGHTS,
) -> float:
    """Evaluate an exponent board without allowing raw tile values to explode."""
    state = np.ascontiguousarray(board, dtype=np.uint8)
    return _evaluate_board_cached(
        state.tobytes(),
        state.shape,
        float(reward),
        _weights_key(weights),
    )
