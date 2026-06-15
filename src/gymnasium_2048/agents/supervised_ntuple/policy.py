from __future__ import annotations

from pathlib import Path

import numpy as np

from gymnasium_2048.agents.expectimax import ExpectimaxPolicy
from gymnasium_2048.agents.supervised_ntuple.model import SupervisedNTupleModel


class SupervisedNTuplePolicy:
    def __init__(
        self,
        model_path: str | Path,
        *,
        depth: int = 0,
        seed: int | None = None,
        chance_samples: int | None = None,
        full_chance_empty_threshold: int = 6,
    ) -> None:
        self.evaluator = SupervisedNTupleModel.load(model_path)
        self.search_policy = ExpectimaxPolicy(
            depth=depth,
            evaluator=self.evaluator,
            seed=seed,
            chance_samples=chance_samples,
            full_chance_empty_threshold=full_chance_empty_threshold,
        )

    @classmethod
    def load(cls, path: str | Path, **kwargs) -> "SupervisedNTuplePolicy":
        return cls(model_path=path, **kwargs)

    def analyze(self, state: np.ndarray):
        return self.search_policy.analyze(state)

    def predict(self, state: np.ndarray) -> int:
        return self.search_policy.predict(state)

    def act(self, observation: np.ndarray, deterministic: bool = True) -> int:
        return self.search_policy.act(observation, deterministic=deterministic)
