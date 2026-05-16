from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gymnasium_2048.agents.heuristic.features import (
    DEFAULT_WEIGHTS,
    HeuristicWeights,
    evaluate_board,
)
from gymnasium_2048.envs import TwentyFortyEightEnv


@dataclass
class HeuristicPolicy:
    weights: HeuristicWeights = DEFAULT_WEIGHTS

    def evaluate(self, state: np.ndarray, action: int) -> float:
       
        next_board, reward, is_legal = TwentyFortyEightEnv.apply_action(
            board=state,
            action=action,
        )

        if not is_legal:
            return -np.inf

        return evaluate_board(
            board=next_board,
            reward=float(reward),
            weights=self.weights,
        )

    def predict(self, state: np.ndarray) -> int:
        action_scores = [self.evaluate(state=state, action=action) for action in range(4)]
        best_score = max(action_scores)

        if best_score == -np.inf:
            return 0

        return int(np.argmax(action_scores))

