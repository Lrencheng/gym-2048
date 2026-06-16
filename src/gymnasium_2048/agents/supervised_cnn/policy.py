from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from gymnasium_2048.agents.expectimax import (
    NUM_ACTIONS,
    NUM_SYMMETRIES,
    ExpectimaxPolicy,
    ExpectimaxResult,
    RootActionValues,
    all_symmetries,
    apply_player_action,
    board_from_observation,
    evaluate_root_actions,
    expand_chance_node,
    masked_softmax,
)
from gymnasium_2048.agents.heuristic.features import RewardTransform, transform_reward
from gymnasium_2048.agents.supervised_cnn.encoding import encode_boards
from gymnasium_2048.agents.supervised_cnn.model import (
    SupervisedCNN,
    config_from_dict,
)


def _torch_load(path: str | Path, map_location: torch.device) -> dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


class CNNAfterstateEvaluator:
    def __init__(
        self,
        model: SupervisedCNN,
        *,
        target_mean: float = 0.0,
        target_std: float = 1.0,
        device: str | torch.device = "cpu",
        symmetry_average: bool = False,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.target_mean = float(target_mean)
        self.target_std = float(target_std)
        self.symmetry_average = bool(symmetry_average)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str = "cpu",
        symmetry_average: bool = False,
    ) -> "CNNAfterstateEvaluator":
        torch_device = torch.device(device)
        payload = _torch_load(path, torch_device)
        model = SupervisedCNN(config_from_dict(payload.get("model_config")))
        model.load_state_dict(payload["model_state_dict"])
        return cls(
            model,
            target_mean=float(payload.get("target_mean", 0.0)),
            target_std=float(payload.get("target_std", 1.0)),
            device=torch_device,
            symmetry_average=symmetry_average,
        )

    def evaluate_afterstate(self, after_board: np.ndarray) -> float:
        return float(self.evaluate_afterstates(after_board)[0])

    def evaluate_afterstates(self, after_boards: np.ndarray) -> np.ndarray:
        raw_boards = np.asarray(after_boards, dtype=np.uint8)
        if raw_boards.ndim == 2:
            raw_boards = raw_boards[None, :, :]
        if len(raw_boards) == 0:
            return np.asarray([], dtype=np.float64)

        boards = (
            np.concatenate([all_symmetries(board) for board in raw_boards], axis=0)
            if self.symmetry_average
            else raw_boards
        )
        encoded = encode_boards(
            boards,
            num_channels=self.model.config.input_channels,
        )
        tensor = torch.from_numpy(encoded).to(self.device)
        with torch.inference_mode():
            normalized = self.model(tensor)
        values = normalized * self.target_std + self.target_mean
        outputs = values.detach().cpu().numpy().astype(np.float64).reshape(-1)
        if self.symmetry_average:
            outputs = outputs.reshape(len(raw_boards), NUM_SYMMETRIES)
            outputs = outputs.mean(axis=1)
        return outputs

    def __call__(self, after_board: np.ndarray) -> float:
        return self.evaluate_afterstate(after_board)


def _evaluate_afterstates(
    evaluator: CNNAfterstateEvaluator,
    afterstates: list[np.ndarray],
) -> np.ndarray:
    if not afterstates:
        return np.asarray([], dtype=np.float64)
    return np.asarray(
        evaluator.evaluate_afterstates(np.asarray(afterstates, dtype=np.uint8)),
        dtype=np.float64,
    )


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


def _evaluate_root_actions_batched(
    board: np.ndarray,
    depth: int,
    evaluator: CNNAfterstateEvaluator,
    *,
    reward_transform: RewardTransform = "raw",
    rng: np.random.Generator | None = None,
    chance_samples: int | None = None,
    full_chance_empty_threshold: int = 6,
) -> RootActionValues:
    if depth < 0:
        raise ValueError("depth must be non-negative")
    if depth > 1:
        return evaluate_root_actions(
            board,
            depth=depth,
            evaluator=evaluator,
            reward_transform=reward_transform,
            rng=rng,
            chance_samples=chance_samples,
            full_chance_empty_threshold=full_chance_empty_threshold,
        )

    state = board_from_observation(board)
    scores = np.full(NUM_ACTIONS, -np.inf, dtype=np.float64)
    legal_mask = np.zeros(NUM_ACTIONS, dtype=bool)
    immediate_rewards = np.zeros(NUM_ACTIONS, dtype=np.float64)
    afterstate_values = np.full(NUM_ACTIONS, np.nan, dtype=np.float64)
    afterstates = np.repeat(state[None, :, :], NUM_ACTIONS, axis=0)

    legal_afterstates: list[np.ndarray] = []
    legal_actions: list[int] = []
    for action in range(NUM_ACTIONS):
        afterstate, reward, is_legal = apply_player_action(state, action)
        afterstates[action] = afterstate
        if not is_legal:
            continue
        legal_mask[action] = True
        immediate_rewards[action] = float(reward)
        legal_actions.append(action)
        legal_afterstates.append(afterstate)

    if depth == 0:
        values = _evaluate_afterstates(evaluator, legal_afterstates)
        for action, value in zip(legal_actions, values):
            afterstate_values[action] = float(value)
            scores[action] = (
                transform_reward(immediate_rewards[action], reward_transform)
                + afterstate_values[action]
            )
        return RootActionValues(
            scores=scores,
            legal_mask=legal_mask,
            immediate_rewards=immediate_rewards,
            afterstate_values=afterstate_values,
            afterstates=afterstates,
        )

    generator = rng or np.random.default_rng()
    unique_leaf_afterstates: list[np.ndarray] = []
    unique_leaf_indices: dict[bytes, int] = {}
    chance_groups: dict[int, list[tuple[float, list[tuple[float, int]]]]] = {}

    for action, root_afterstate in zip(legal_actions, legal_afterstates):
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
                    (
                        transform_reward(float(reward), reward_transform),
                        leaf_index,
                    )
                )
            action_groups.append((outcome.probability, candidates))
        chance_groups[action] = action_groups

    leaf_values = _evaluate_afterstates(evaluator, unique_leaf_afterstates)
    for action, action_groups in chance_groups.items():
        value = 0.0
        for probability, candidates in action_groups:
            if candidates:
                player_value = max(
                    reward + float(leaf_values[leaf_index])
                    for reward, leaf_index in candidates
                )
            else:
                player_value = 0.0
            value += probability * player_value
        afterstate_values[action] = float(value)
        scores[action] = (
            transform_reward(immediate_rewards[action], reward_transform)
            + afterstate_values[action]
        )

    return RootActionValues(
        scores=scores,
        legal_mask=legal_mask,
        immediate_rewards=immediate_rewards,
        afterstate_values=afterstate_values,
        afterstates=afterstates,
    )


class SupervisedCNNPolicy:
    def __init__(
        self,
        checkpoint: str | Path,
        *,
        depth: int = 0,
        device: str = "cpu",
        seed: int | None = None,
        chance_samples: int | None = None,
        full_chance_empty_threshold: int = 6,
        symmetry_average: bool = False,
    ) -> None:
        self.evaluator = CNNAfterstateEvaluator.load(
            checkpoint,
            device=device,
            symmetry_average=symmetry_average,
        )
        self.search_policy = ExpectimaxPolicy(
            depth=depth,
            evaluator=self.evaluator,
            seed=seed,
            chance_samples=chance_samples,
            full_chance_empty_threshold=full_chance_empty_threshold,
        )

    @classmethod
    def load(cls, path: str | Path, **kwargs) -> "SupervisedCNNPolicy":
        return cls(checkpoint=path, **kwargs)

    def analyze(self, state: np.ndarray) -> ExpectimaxResult:
        result = _evaluate_root_actions_batched(
            board_from_observation(state),
            depth=self.search_policy.depth,
            evaluator=self.evaluator,
            reward_transform=self.search_policy.reward_transform,
            rng=self.search_policy.rng,
            chance_samples=self.search_policy.chance_samples,
            full_chance_empty_threshold=self.search_policy.full_chance_empty_threshold,
        )
        probabilities = masked_softmax(
            result.scores,
            result.legal_mask,
            temperature=self.search_policy.temperature,
        )
        action = int(np.argmax(result.scores)) if np.any(result.legal_mask) else 0
        return ExpectimaxResult(
            action=action,
            scores=result.scores,
            legal_mask=result.legal_mask,
            probabilities=probabilities,
        )

    def predict(self, state: np.ndarray) -> int:
        return self.analyze(state=state).action

    def act(self, observation: np.ndarray, deterministic: bool = True) -> int:
        result = self.analyze(state=observation)
        if deterministic or not np.any(result.legal_mask):
            return result.action
        return int(self.search_policy.rng.choice(NUM_ACTIONS, p=result.probabilities))
