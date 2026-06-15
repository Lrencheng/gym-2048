from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from gymnasium_2048.agents.expectimax import ExpectimaxPolicy, all_symmetries
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
        boards = (
            all_symmetries(after_board)
            if self.symmetry_average
            else np.asarray(after_board, dtype=np.uint8)[None, :, :]
        )
        encoded = encode_boards(
            boards,
            num_channels=self.model.config.input_channels,
        )
        tensor = torch.from_numpy(encoded).to(self.device)
        with torch.no_grad():
            normalized = self.model(tensor)
        values = normalized * self.target_std + self.target_mean
        return float(values.mean().cpu())

    def __call__(self, after_board: np.ndarray) -> float:
        return self.evaluate_afterstate(after_board)


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

    def analyze(self, state: np.ndarray):
        return self.search_policy.analyze(state)

    def predict(self, state: np.ndarray) -> int:
        return self.search_policy.predict(state)

    def act(self, observation: np.ndarray, deterministic: bool = True) -> int:
        return self.search_policy.act(observation, deterministic=deterministic)
