from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gymnasium_2048.agents.expectimax.board import (
    NUM_ACTIONS,
    board_from_observation,
    masked_softmax,
)
from gymnasium_2048.agents.expectimax.heuristic import (
    DEFAULT_WEIGHTS,
    ExpectimaxHeuristicWeights,
    evaluate_board,
)
from gymnasium_2048.agents.expectimax.fast_move import fast_apply_action
from gymnasium_2048.agents.heuristic.features import RewardTransform, transform_reward
from gymnasium_2048.envs import TwentyFortyEightEnv


@dataclass(frozen=True)
class ExpectimaxResult:
    action: int
    scores: np.ndarray
    legal_mask: np.ndarray
    probabilities: np.ndarray


class ExpectimaxPolicy:
    def __init__(
        self,
        depth: int = 2,
        temperature: float = 1.0,
        weights: ExpectimaxHeuristicWeights = DEFAULT_WEIGHTS,
        reward_transform: RewardTransform = "log2p1",
        seed: int | None = None,
        chance_samples: int | None = None,
        full_chance_empty_threshold: int = 6,
    ) -> None:
        if depth < 1:
            raise ValueError("depth must be at least 1")
        if chance_samples is not None and chance_samples < 1:
            raise ValueError("chance_samples must be positive when provided")
        if full_chance_empty_threshold < 0:
            raise ValueError("full_chance_empty_threshold must be non-negative")
        self.depth = int(depth)
        self.temperature = float(temperature)
        self.weights = weights
        self.reward_transform = reward_transform
        self.chance_samples = chance_samples
        self.full_chance_empty_threshold = int(full_chance_empty_threshold)
        self.rng = np.random.default_rng(seed)
        self._player_cache: dict[tuple[bytes, tuple[int, int], int], float] = {}
        self._chance_cache: dict[tuple[bytes, tuple[int, int], int], float] = {}

    def evaluate(self, state: np.ndarray, action: int) -> float:
        board = board_from_observation(state)
        return self._after_action_value(board=board, action=action, remaining_depth=self.depth)

    def analyze(
        self,
        state: np.ndarray,
    ) -> ExpectimaxResult:
        board = board_from_observation(state)
        self._player_cache.clear()
        self._chance_cache.clear()

        mask = np.zeros(NUM_ACTIONS, dtype=bool)
        scores = np.full(NUM_ACTIONS, -np.inf, dtype=np.float64)
        for action in range(NUM_ACTIONS):
            next_board, reward, is_legal = fast_apply_action(board=board, action=action)
            if not is_legal:
                continue
            mask[action] = True
            scores[action] = self._after_action_value_from_transition(
                next_board=next_board,
                reward=reward,
                remaining_depth=self.depth,
            )

        probabilities = masked_softmax(scores, mask, temperature=self.temperature)
        action = int(np.argmax(scores)) if np.any(mask) else 0
        return ExpectimaxResult(
            action=action,
            scores=scores,
            legal_mask=mask,
            probabilities=probabilities,
        )

    def predict(self, state: np.ndarray) -> int:
        return self.analyze(state=state).action

    def act(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> int:
        result = self.analyze(state=observation)
        if deterministic or not np.any(result.legal_mask):
            return result.action
        return int(self.rng.choice(NUM_ACTIONS, p=result.probabilities))

    @staticmethod
    def _cache_key(board: np.ndarray, remaining_depth: int) -> tuple[bytes, tuple[int, int], int]:
        state = np.ascontiguousarray(board, dtype=np.uint8)
        return state.tobytes(), state.shape, remaining_depth

    def _after_action_value(
        self,
        board: np.ndarray,
        action: int,
        remaining_depth: int,
    ) -> float:
        next_board, reward, is_legal = fast_apply_action(
            board=board,
            action=action,
        )
        if not is_legal:
            return -np.inf

        return self._after_action_value_from_transition(
            next_board=next_board,
            reward=reward,
            remaining_depth=remaining_depth,
        )

    def _after_action_value_from_transition(
        self,
        next_board: np.ndarray,
        reward: int,
        remaining_depth: int,
    ) -> float:
        if remaining_depth <= 1 or TwentyFortyEightEnv.is_terminated(next_board):
            return evaluate_board(
                next_board,
                reward=float(reward),
                weights=self.weights,
                reward_transform=self.reward_transform,
            )

        immediate_reward = transform_reward(float(reward), self.reward_transform)
        return float(
            self.weights.reward * immediate_reward
            + self._chance_value(next_board, remaining_depth - 1)
        )

    def _player_value(self, board: np.ndarray, remaining_depth: int) -> float:
        key = self._cache_key(board, remaining_depth)
        cached = self._player_cache.get(key)
        if cached is not None:
            return cached

        legal_values = [
            self._after_action_value(board=board, action=action, remaining_depth=remaining_depth)
            for action in range(NUM_ACTIONS)
        ]
        value = float(np.max(legal_values))
        if not np.isfinite(value):
            value = evaluate_board(
                board,
                reward=0.0,
                weights=self.weights,
                reward_transform=self.reward_transform,
            )

        self._player_cache[key] = value
        return value

    def _chance_value(self, board: np.ndarray, remaining_depth: int) -> float:
        key = self._cache_key(board, remaining_depth)
        cached = self._chance_cache.get(key)
        if cached is not None:
            return cached

        empty_cells = np.argwhere(board == 0)
        if empty_cells.size == 0:
            value = self._player_value(board, remaining_depth)
            self._chance_cache[key] = value
            return value

        sampled_empty_cells = self._sample_empty_cells(empty_cells)
        total = 0.0
        outcome_probability = 1.0 / float(len(sampled_empty_cells))
        for row, col in sampled_empty_cells:
            for tile_exponent, tile_probability in ((1, 0.9), (2, 0.1)):
                spawned = board.copy()
                spawned[row, col] = tile_exponent
                total += (
                    outcome_probability
                    * tile_probability
                    * self._player_value(spawned, remaining_depth)
                )

        value = float(total)
        self._chance_cache[key] = value
        return value

    def _sample_empty_cells(self, empty_cells: np.ndarray) -> np.ndarray:
        if (
            self.chance_samples is None
            or len(empty_cells) <= self.full_chance_empty_threshold
            or len(empty_cells) <= self.chance_samples
        ):
            return empty_cells

        indices = self.rng.choice(
            len(empty_cells),
            size=self.chance_samples,
            replace=False,
        )
        return empty_cells[np.sort(indices)]
