from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from gymnasium_2048.agents.expectimax.board import (
    NUM_ACTIONS,
    board_from_observation,
    normalize_legal_mask,
)
from gymnasium_2048.agents.supervised_cnn.encoding import encode_board
from gymnasium_2048.agents.supervised_cnn.model import SupervisedCNN, config_from_dict


@dataclass(frozen=True)
class SupervisedCNNResult:
    action: int
    logits: np.ndarray
    legal_mask: np.ndarray
    probabilities: np.ndarray


def _torch_load(path: str | Path, map_location: torch.device) -> dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


class SupervisedCNNPolicy:
    def __init__(
        self,
        checkpoint: str | Path,
        device: str = "cpu",
        seed: int | None = None,
    ) -> None:
        self.device = torch.device(device)
        payload = _torch_load(checkpoint, map_location=self.device)
        self.config = config_from_dict(payload.get("model_config"))
        self.model = SupervisedCNN(self.config).to(self.device)
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.eval()
        self.rng = np.random.default_rng(seed)

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: str = "cpu",
        seed: int | None = None,
    ) -> "SupervisedCNNPolicy":
        return cls(checkpoint=path, device=device, seed=seed)

    def analyze(
        self,
        state: np.ndarray,
        legal_mask: np.ndarray | None = None,
    ) -> SupervisedCNNResult:
        board = board_from_observation(state)
        mask = normalize_legal_mask(board, legal_mask)

        encoded = encode_board(board, num_channels=self.config.input_channels)
        tensor = torch.from_numpy(encoded).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor).squeeze(0).cpu().numpy().astype(np.float64)

        if not np.any(mask):
            return SupervisedCNNResult(
                action=0,
                logits=logits,
                legal_mask=mask,
                probabilities=np.zeros(NUM_ACTIONS, dtype=np.float64),
            )

        masked_logits = np.where(mask, logits, -1.0e9)
        probabilities = F.softmax(torch.from_numpy(masked_logits), dim=-1).numpy()
        return SupervisedCNNResult(
            action=int(np.argmax(masked_logits)),
            logits=logits,
            legal_mask=mask,
            probabilities=probabilities,
        )

    def predict(self, state: np.ndarray, legal_mask: np.ndarray | None = None) -> int:
        return self.analyze(state=state, legal_mask=legal_mask).action

    def act(
        self,
        observation: np.ndarray,
        legal_mask: np.ndarray | None = None,
        deterministic: bool = True,
    ) -> int:
        result = self.analyze(state=observation, legal_mask=legal_mask)
        if deterministic or not np.any(result.legal_mask):
            return result.action
        return int(self.rng.choice(NUM_ACTIONS, p=result.probabilities))
