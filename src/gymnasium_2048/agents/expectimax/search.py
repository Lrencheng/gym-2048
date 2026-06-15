from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from gymnasium_2048.agents.expectimax.board import (
    NUM_ACTIONS,
    board_from_observation,
)
from gymnasium_2048.agents.expectimax.fast_move import fast_apply_action
from gymnasium_2048.agents.heuristic.features import RewardTransform, transform_reward

AfterstateEvaluator = Callable[[np.ndarray], float]


@dataclass(frozen=True)
class ChanceOutcome:
    board: np.ndarray
    probability: float


@dataclass(frozen=True)
class RootActionValues:
    scores: np.ndarray
    legal_mask: np.ndarray
    immediate_rewards: np.ndarray
    afterstate_values: np.ndarray
    afterstates: np.ndarray


def apply_player_action(
    board: np.ndarray,
    action: int,
) -> tuple[np.ndarray, int, bool]:
    """Apply a player move without spawning a random tile."""
    state = board_from_observation(board)
    return fast_apply_action(state, int(action))


def _sample_empty_cells(
    empty_cells: np.ndarray,
    rng: np.random.Generator,
    chance_samples: int | None,
    full_chance_empty_threshold: int,
) -> np.ndarray:
    if (
        chance_samples is None
        or len(empty_cells) <= full_chance_empty_threshold
        or len(empty_cells) <= chance_samples
    ):
        return empty_cells
    indices = rng.choice(len(empty_cells), size=chance_samples, replace=False)
    return empty_cells[np.sort(indices)]


def expand_chance_node(
    after_board: np.ndarray,
    *,
    rng: np.random.Generator | None = None,
    chance_samples: int | None = None,
    full_chance_empty_threshold: int = 6,
) -> list[ChanceOutcome]:
    """Expand random 2/4 spawns after a player move.

    Sampling approximates the uniform empty-cell expectation while preserving
    a probability sum of one over the sampled outcomes.
    """
    if chance_samples is not None and chance_samples < 1:
        raise ValueError("chance_samples must be positive when provided")
    if full_chance_empty_threshold < 0:
        raise ValueError("full_chance_empty_threshold must be non-negative")

    board = board_from_observation(after_board)
    empty_cells = np.argwhere(board == 0)
    if len(empty_cells) == 0:
        return [ChanceOutcome(board=board.copy(), probability=1.0)]

    generator = rng or np.random.default_rng()
    selected = _sample_empty_cells(
        empty_cells,
        generator,
        chance_samples,
        full_chance_empty_threshold,
    )
    cell_probability = 1.0 / float(len(selected))
    outcomes: list[ChanceOutcome] = []
    for row, col in selected:
        for exponent, tile_probability in ((1, 0.9), (2, 0.1)):
            spawned = board.copy()
            spawned[row, col] = exponent
            outcomes.append(
                ChanceOutcome(
                    board=spawned,
                    probability=cell_probability * tile_probability,
                )
            )
    return outcomes


class _Search:
    def __init__(
        self,
        evaluator: AfterstateEvaluator,
        reward_transform: RewardTransform,
        rng: np.random.Generator,
        chance_samples: int | None,
        full_chance_empty_threshold: int,
    ) -> None:
        self.evaluator = evaluator
        self.reward_transform = reward_transform
        self.rng = rng
        self.chance_samples = chance_samples
        self.full_chance_empty_threshold = full_chance_empty_threshold
        self._afterstate_cache: dict[tuple[bytes, int], float] = {}
        self._player_cache: dict[tuple[bytes, int], float] = {}

    @staticmethod
    def _key(board: np.ndarray, depth: int) -> tuple[bytes, int]:
        state = np.ascontiguousarray(board, dtype=np.uint8)
        return state.tobytes(), int(depth)

    def afterstate_value(self, after_board: np.ndarray, depth: int) -> float:
        if depth < 0:
            raise ValueError("depth must be non-negative")
        board = board_from_observation(after_board)
        if depth == 0:
            return float(self.evaluator(board))

        key = self._key(board, depth)
        if key in self._afterstate_cache:
            return self._afterstate_cache[key]

        value = float(
            sum(
                outcome.probability * self.player_value(outcome.board, depth)
                for outcome in expand_chance_node(
                    board,
                    rng=self.rng,
                    chance_samples=self.chance_samples,
                    full_chance_empty_threshold=self.full_chance_empty_threshold,
                )
            )
        )
        self._afterstate_cache[key] = value
        return value

    def player_value(self, board: np.ndarray, depth: int) -> float:
        if depth < 1:
            raise ValueError("player-state depth must be at least one")
        state = board_from_observation(board)
        key = self._key(state, depth)
        if key in self._player_cache:
            return self._player_cache[key]

        best = -np.inf
        for action in range(NUM_ACTIONS):
            afterstate, reward, is_legal = apply_player_action(state, action)
            if not is_legal:
                continue
            value = transform_reward(float(reward), self.reward_transform)
            value += self.afterstate_value(afterstate, depth - 1)
            best = max(best, value)

        value = 0.0 if not np.isfinite(best) else float(best)
        self._player_cache[key] = value
        return value

    def root_values(self, board: np.ndarray, depth: int) -> RootActionValues:
        if depth < 0:
            raise ValueError("depth must be non-negative")
        state = board_from_observation(board)
        scores = np.full(NUM_ACTIONS, -np.inf, dtype=np.float64)
        legal_mask = np.zeros(NUM_ACTIONS, dtype=bool)
        immediate_rewards = np.zeros(NUM_ACTIONS, dtype=np.float64)
        afterstate_values = np.full(NUM_ACTIONS, np.nan, dtype=np.float64)
        afterstates = np.repeat(state[None, :, :], NUM_ACTIONS, axis=0)

        for action in range(NUM_ACTIONS):
            afterstate, reward, is_legal = apply_player_action(state, action)
            afterstates[action] = afterstate
            if not is_legal:
                continue
            legal_mask[action] = True
            immediate_rewards[action] = float(reward)
            afterstate_values[action] = self.afterstate_value(afterstate, depth)
            scores[action] = (
                transform_reward(float(reward), self.reward_transform)
                + afterstate_values[action]
            )

        return RootActionValues(
            scores=scores,
            legal_mask=legal_mask,
            immediate_rewards=immediate_rewards,
            afterstate_values=afterstate_values,
            afterstates=afterstates,
        )


def _make_search(
    evaluator: AfterstateEvaluator,
    reward_transform: RewardTransform,
    rng: np.random.Generator | None,
    chance_samples: int | None,
    full_chance_empty_threshold: int,
) -> _Search:
    return _Search(
        evaluator=evaluator,
        reward_transform=reward_transform,
        rng=rng or np.random.default_rng(),
        chance_samples=chance_samples,
        full_chance_empty_threshold=full_chance_empty_threshold,
    )


def expectimax_afterstate_value(
    after_board: np.ndarray,
    depth: int,
    evaluator: AfterstateEvaluator,
    *,
    reward_transform: RewardTransform = "raw",
    rng: np.random.Generator | None = None,
    chance_samples: int | None = None,
    full_chance_empty_threshold: int = 6,
) -> float:
    """Evaluate an afterstate.

    Depth is the number of future player decisions after the random tile spawn.
    At depth zero, the evaluator is called directly on the afterstate.
    """
    search = _make_search(
        evaluator,
        reward_transform,
        rng,
        chance_samples,
        full_chance_empty_threshold,
    )
    return search.afterstate_value(after_board, depth)


def player_state_value(
    board: np.ndarray,
    depth: int,
    evaluator: AfterstateEvaluator,
    *,
    reward_transform: RewardTransform = "raw",
    rng: np.random.Generator | None = None,
    chance_samples: int | None = None,
    full_chance_empty_threshold: int = 6,
) -> float:
    search = _make_search(
        evaluator,
        reward_transform,
        rng,
        chance_samples,
        full_chance_empty_threshold,
    )
    return search.player_value(board, depth)


def evaluate_root_actions(
    board: np.ndarray,
    depth: int,
    evaluator: AfterstateEvaluator,
    *,
    reward_transform: RewardTransform = "raw",
    rng: np.random.Generator | None = None,
    chance_samples: int | None = None,
    full_chance_empty_threshold: int = 6,
) -> RootActionValues:
    search = _make_search(
        evaluator,
        reward_transform,
        rng,
        chance_samples,
        full_chance_empty_threshold,
    )
    return search.root_values(board, depth)


def select_best_action(
    board: np.ndarray,
    depth: int,
    evaluator: AfterstateEvaluator,
    **kwargs,
) -> int:
    result = evaluate_root_actions(board, depth, evaluator, **kwargs)
    return int(np.argmax(result.scores)) if np.any(result.legal_mask) else 0
