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
from gymnasium_2048.agents.expectimax.search import (
    AfterstateEvaluator,
    evaluate_root_actions,
)
from gymnasium_2048.agents.heuristic.features import RewardTransform


@dataclass(frozen=True)
class ExpectimaxResult:
    action: int
    scores: np.ndarray
    legal_mask: np.ndarray
    probabilities: np.ndarray


class ExpectimaxPolicy:
    """Expectimax policy using an injectable afterstate leaf evaluator."""

    def __init__(
        self,
        depth: int = 2,
        temperature: float = 1.0,
        weights: ExpectimaxHeuristicWeights = DEFAULT_WEIGHTS,
        reward_transform: RewardTransform = "raw",
        seed: int | None = None,
        chance_samples: int | None = None,
        full_chance_empty_threshold: int = 6,
        evaluator: AfterstateEvaluator | None = None,
    ) -> None:
        if depth < 0:
            raise ValueError("depth must be non-negative")
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
        self.evaluator = evaluator or (
            lambda after_board: evaluate_board(
                after_board,
                reward=0.0,
                weights=self.weights,
                reward_transform=self.reward_transform,
            )
        )

    def evaluate(self, state: np.ndarray, action: int) -> float:
        result = evaluate_root_actions(
            board_from_observation(state),
            depth=self.depth,
            evaluator=self.evaluator,
            reward_transform=self.reward_transform,
            rng=self.rng,
            chance_samples=self.chance_samples,
            full_chance_empty_threshold=self.full_chance_empty_threshold,
        )
        return float(result.scores[int(action)])

    def analyze(self, state: np.ndarray) -> ExpectimaxResult:
        result = evaluate_root_actions(
            board_from_observation(state),
            depth=self.depth,
            evaluator=self.evaluator,
            reward_transform=self.reward_transform,
            rng=self.rng,
            chance_samples=self.chance_samples,
            full_chance_empty_threshold=self.full_chance_empty_threshold,
        )
        probabilities = masked_softmax(
            result.scores,
            result.legal_mask,
            temperature=self.temperature,
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

    def act(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> int:
        result = self.analyze(state=observation)
        if deterministic or not np.any(result.legal_mask):
            return result.action
        return int(self.rng.choice(NUM_ACTIONS, p=result.probabilities))

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
