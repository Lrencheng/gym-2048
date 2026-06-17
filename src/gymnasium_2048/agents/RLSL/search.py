from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gymnasium_2048.agents.expectimax import (
    AfterstateEvaluator,
    NUM_ACTIONS,
    RootActionValues,
    apply_player_action,
    board_from_observation,
    expand_chance_node,
)
from gymnasium_2048.agents.heuristic.features import RewardTransform, transform_reward


@dataclass(frozen=True)
class SearchImprovedActionSample:
    action: int
    afterstate: np.ndarray
    target_value: float
    root_score: float
    immediate_reward: float


@dataclass(frozen=True)
class SearchImprovedDecision:
    action: int
    afterstate: np.ndarray
    target_value: float
    root_score: float
    immediate_reward: float
    scores: np.ndarray
    legal_mask: np.ndarray
    action_samples: tuple[SearchImprovedActionSample, ...]


def _validate_search_depth(search_depth: int) -> int:
    depth = int(search_depth)
    if depth != 1:
        raise ValueError("RLSL currently supports only search_depth=1")
    return depth


def _evaluate_afterstates(
    evaluator: AfterstateEvaluator,
    afterstates: list[np.ndarray],
) -> np.ndarray:
    if not afterstates:
        return np.asarray([], dtype=np.float64)
    boards = np.asarray(afterstates, dtype=np.uint8)
    batch_evaluator = getattr(evaluator, "evaluate_afterstates", None)
    if callable(batch_evaluator):
        return np.asarray(batch_evaluator(boards), dtype=np.float64).reshape(-1)
    return np.asarray([float(evaluator(board)) for board in boards], dtype=np.float64)


def _append_unique_afterstate(
    afterstate: np.ndarray,
    unique_afterstates: list[np.ndarray],
    unique_indices: dict[bytes, int],
) -> int:
    state = np.ascontiguousarray(afterstate, dtype=np.uint8)
    key = state.tobytes()
    index = unique_indices.get(key)
    if index is None:
        index = len(unique_afterstates)
        unique_indices[key] = index
        unique_afterstates.append(state.copy())
    return index


def _evaluate_depth_one_chance_groups(
    chance_groups: dict[int, list[tuple[float, list[tuple[float, int]]]]],
    leaf_values: np.ndarray,
) -> dict[int, float]:
    values: dict[int, float] = {}
    for group_id, groups in chance_groups.items():
        value = 0.0
        for probability, candidates in groups:
            if candidates:
                player_value = max(
                    reward + float(leaf_values[leaf_index])
                    for reward, leaf_index in candidates
                )
            else:
                player_value = 0.0
            value += probability * player_value
        values[group_id] = float(value)
    return values


def _depth_one_afterstate_value_batched(
    afterstate: np.ndarray,
    evaluator: AfterstateEvaluator,
    *,
    reward_transform: RewardTransform,
    rng: np.random.Generator | None,
    chance_samples: int | None,
    full_chance_empty_threshold: int,
) -> float:
    board = board_from_observation(afterstate)
    generator = rng or np.random.default_rng()
    unique_leaf_afterstates: list[np.ndarray] = []
    unique_leaf_indices: dict[bytes, int] = {}
    chance_groups: dict[int, list[tuple[float, list[tuple[float, int]]]]] = {0: []}

    for outcome in expand_chance_node(
        board,
        rng=generator,
        chance_samples=chance_samples,
        full_chance_empty_threshold=full_chance_empty_threshold,
    ):
        candidates = []
        for next_action in range(NUM_ACTIONS):
            leaf_afterstate, reward, is_legal = apply_player_action(
                outcome.board,
                next_action,
            )
            if not is_legal:
                continue
            leaf_index = _append_unique_afterstate(
                leaf_afterstate,
                unique_leaf_afterstates,
                unique_leaf_indices,
            )
            candidates.append(
                (transform_reward(float(reward), reward_transform), leaf_index)
            )
        chance_groups[0].append((outcome.probability, candidates))

    leaf_values = _evaluate_afterstates(evaluator, unique_leaf_afterstates)
    return _evaluate_depth_one_chance_groups(chance_groups, leaf_values)[0]


def _evaluate_root_actions_batched_depth_one(
    board: np.ndarray,
    evaluator: AfterstateEvaluator,
    *,
    reward_transform: RewardTransform,
    rng: np.random.Generator | None,
    chance_samples: int | None,
    full_chance_empty_threshold: int,
) -> RootActionValues:
    state = board_from_observation(board)
    scores = np.full(NUM_ACTIONS, -np.inf, dtype=np.float64)
    legal_mask = np.zeros(NUM_ACTIONS, dtype=bool)
    immediate_rewards = np.zeros(NUM_ACTIONS, dtype=np.float64)
    afterstate_values = np.full(NUM_ACTIONS, np.nan, dtype=np.float64)
    afterstates = np.repeat(state[None, :, :], NUM_ACTIONS, axis=0)

    legal_afterstates: list[tuple[int, np.ndarray]] = []
    for action in range(NUM_ACTIONS):
        afterstate, reward, is_legal = apply_player_action(state, action)
        afterstates[action] = afterstate
        if not is_legal:
            continue
        legal_mask[action] = True
        immediate_rewards[action] = float(reward)
        legal_afterstates.append((action, afterstate))

    generator = rng or np.random.default_rng()
    unique_leaf_afterstates: list[np.ndarray] = []
    unique_leaf_indices: dict[bytes, int] = {}
    chance_groups: dict[int, list[tuple[float, list[tuple[float, int]]]]] = {}

    for action, root_afterstate in legal_afterstates:
        action_groups = []
        for outcome in expand_chance_node(
            root_afterstate,
            rng=generator,
            chance_samples=chance_samples,
            full_chance_empty_threshold=full_chance_empty_threshold,
        ):
            candidates = []
            for next_action in range(NUM_ACTIONS):
                leaf_afterstate, reward, is_legal = apply_player_action(
                    outcome.board,
                    next_action,
                )
                if not is_legal:
                    continue
                leaf_index = _append_unique_afterstate(
                    leaf_afterstate,
                    unique_leaf_afterstates,
                    unique_leaf_indices,
                )
                candidates.append(
                    (transform_reward(float(reward), reward_transform), leaf_index)
                )
            action_groups.append((outcome.probability, candidates))
        chance_groups[action] = action_groups

    leaf_values = _evaluate_afterstates(evaluator, unique_leaf_afterstates)
    values = _evaluate_depth_one_chance_groups(chance_groups, leaf_values)
    for action, value in values.items():
        afterstate_values[action] = value
        scores[action] = (
            transform_reward(immediate_rewards[action], reward_transform) + value
        )

    return RootActionValues(
        scores=scores,
        legal_mask=legal_mask,
        immediate_rewards=immediate_rewards,
        afterstate_values=afterstate_values,
        afterstates=afterstates,
    )


def search_improved_afterstate_value(
    afterstate: np.ndarray,
    evaluator: AfterstateEvaluator,
    *,
    search_depth: int = 1,
    reward_transform: RewardTransform = "raw",
    rng: np.random.Generator | None = None,
    chance_samples: int | None = None,
    full_chance_empty_threshold: int = 6,
) -> float:
    """Return the depth-1 search-improved continuation value for an afterstate."""
    _validate_search_depth(search_depth)
    return _depth_one_afterstate_value_batched(
        afterstate,
        evaluator=evaluator,
        reward_transform=reward_transform,
        rng=rng,
        chance_samples=chance_samples,
        full_chance_empty_threshold=full_chance_empty_threshold,
    )


def choose_search_improved_action(
    board: np.ndarray,
    evaluator: AfterstateEvaluator,
    *,
    search_depth: int = 1,
    reward_transform: RewardTransform = "raw",
    rng: np.random.Generator | None = None,
    chance_samples: int | None = None,
    full_chance_empty_threshold: int = 6,
) -> SearchImprovedDecision:
    """Choose a root action by reward + search-improved afterstate value.

    The returned target_value is only U_search(afterstate). It intentionally
    excludes the root immediate reward used for action selection.
    """
    _validate_search_depth(search_depth)
    root_values: RootActionValues = _evaluate_root_actions_batched_depth_one(
        board_from_observation(board),
        evaluator,
        reward_transform=reward_transform,
        rng=rng,
        chance_samples=chance_samples,
        full_chance_empty_threshold=full_chance_empty_threshold,
    )
    if not np.any(root_values.legal_mask):
        return SearchImprovedDecision(
            action=0,
            afterstate=board_from_observation(board),
            target_value=0.0,
            root_score=0.0,
            immediate_reward=0.0,
            scores=root_values.scores.copy(),
            legal_mask=root_values.legal_mask.copy(),
            action_samples=(),
        )

    action = int(np.argmax(root_values.scores))
    action_samples = tuple(
        SearchImprovedActionSample(
            action=int(candidate_action),
            afterstate=root_values.afterstates[int(candidate_action)].copy(),
            target_value=float(root_values.afterstate_values[int(candidate_action)]),
            root_score=float(root_values.scores[int(candidate_action)]),
            immediate_reward=float(root_values.immediate_rewards[int(candidate_action)]),
        )
        for candidate_action in np.flatnonzero(root_values.legal_mask)
    )
    return SearchImprovedDecision(
        action=action,
        afterstate=root_values.afterstates[action].copy(),
        target_value=float(root_values.afterstate_values[action]),
        root_score=float(root_values.scores[action]),
        immediate_reward=float(root_values.immediate_rewards[action]),
        scores=root_values.scores.copy(),
        legal_mask=root_values.legal_mask.copy(),
        action_samples=action_samples,
    )
