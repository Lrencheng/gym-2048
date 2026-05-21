from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from gymnasium_2048.agents.expectimax.data import load_expectimax_dataset
from gymnasium_2048.agents.supervised_cnn.encoding import encode_board


@dataclass(frozen=True)
class WeightConfig:
    enabled: bool = True
    score_weight: float = 0.5
    tile_weight: float = 0.4
    late_game_weight: float = 0.2
    difficulty_weight: float = 0.2
    max_weight: float = 3.0


def compute_sample_weights(
    final_scores: np.ndarray,
    final_max_tiles: np.ndarray,
    steps: np.ndarray,
    empty_counts: np.ndarray,
    config: WeightConfig = WeightConfig(),
) -> np.ndarray:
    if not config.enabled:
        return np.ones_like(final_scores, dtype=np.float32)

    final_scores = np.asarray(final_scores, dtype=np.float64)
    final_max_tiles = np.asarray(final_max_tiles, dtype=np.float64)
    steps = np.asarray(steps, dtype=np.float64)
    empty_counts = np.asarray(empty_counts, dtype=np.float64)

    score_denom = np.log1p(max(float(np.max(final_scores)), 1.0))
    tile_denom = np.log2(max(float(np.max(final_max_tiles)), 2.0))
    step_denom = max(float(np.max(steps)), 1.0)

    score_component = np.log1p(final_scores) / score_denom
    tile_component = np.log2(np.maximum(final_max_tiles, 2.0)) / tile_denom
    late_component = steps / step_denom
    difficulty_component = np.clip((16.0 - empty_counts) / 16.0, 0.0, 1.0)

    weights = (
        1.0
        + config.score_weight * score_component
        + config.tile_weight * tile_component
        + config.late_game_weight * late_component
        + config.difficulty_weight * difficulty_component
    )
    weights = np.clip(weights, 1.0, config.max_weight)
    mean_weight = float(np.mean(weights))
    if mean_weight > 0.0:
        weights = weights / mean_weight
    return weights.astype(np.float32)


class ExpectimaxDataset(Dataset):
    def __init__(
        self,
        path: str | Path | None = None,
        indices: np.ndarray | None = None,
        num_channels: int = 16,
        weight_config: WeightConfig = WeightConfig(),
        dataset: dict[str, np.ndarray | dict[str, Any]] | None = None,
        weights: np.ndarray | None = None,
        encode_boards: bool = True,
    ) -> None:
        if dataset is None:
            if path is None:
                raise ValueError("path is required when dataset is not provided")
            dataset = load_expectimax_dataset(path)

        self.boards = np.asarray(dataset["boards"], dtype=np.uint8)
        self.legal_masks = np.asarray(dataset["legal_masks"], dtype=bool)
        self.action_probs = np.asarray(dataset["action_probs"], dtype=np.float32)
        self.actions = np.asarray(dataset["actions"], dtype=np.int64)
        self.final_scores = np.asarray(dataset["final_scores"], dtype=np.int64)
        self.final_max_tiles = np.asarray(dataset["final_max_tiles"], dtype=np.int64)
        self.empty_counts = np.asarray(dataset["empty_counts"], dtype=np.int64)
        self.steps = np.asarray(dataset["steps"], dtype=np.int64)
        self.weights = (
            np.asarray(weights, dtype=np.float32)
            if weights is not None
            else compute_sample_weights(
                final_scores=self.final_scores,
                final_max_tiles=self.final_max_tiles,
                steps=self.steps,
                empty_counts=self.empty_counts,
                config=weight_config,
            )
        )
        self.num_channels = num_channels
        self.encode_boards = encode_boards
        self.indices = (
            np.arange(len(self.boards), dtype=np.int64)
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )

    def subset(
        self,
        indices: np.ndarray,
        encode_boards: bool | None = None,
    ) -> "ExpectimaxDataset":
        subset = object.__new__(ExpectimaxDataset)
        subset.boards = self.boards
        subset.legal_masks = self.legal_masks
        subset.action_probs = self.action_probs
        subset.actions = self.actions
        subset.final_scores = self.final_scores
        subset.final_max_tiles = self.final_max_tiles
        subset.empty_counts = self.empty_counts
        subset.steps = self.steps
        subset.weights = self.weights
        subset.num_channels = self.num_channels
        subset.encode_boards = self.encode_boards if encode_boards is None else encode_boards
        subset.indices = np.asarray(indices, dtype=np.int64)
        return subset

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        data_index = int(self.indices[index])
        board = self.boards[data_index]
        model_input = (
            torch.from_numpy(encode_board(board, num_channels=self.num_channels))
            if self.encode_boards
            else torch.from_numpy(board)
        )
        return (
            model_input,
            torch.from_numpy(self.legal_masks[data_index].astype(np.float32)),
            torch.from_numpy(self.action_probs[data_index]),
            torch.tensor(self.actions[data_index], dtype=torch.long),
            torch.tensor(self.weights[data_index], dtype=torch.float32),
        )


def split_indices(
    n_samples: int,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    if n_samples < 1:
        raise ValueError("dataset must contain at least one sample")
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples, dtype=np.int64)
    rng.shuffle(indices)
    val_size = int(round(n_samples * validation_fraction))
    val_size = min(max(val_size, 1 if n_samples > 1 else 0), n_samples - 1 if n_samples > 1 else 0)
    return indices[val_size:], indices[:val_size]
