from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from gymnasium_2048.agents.expectimax.data import load_expectimax_dataset
from gymnasium_2048.agents.expectimax.symmetry import apply_symmetry
from gymnasium_2048.agents.supervised_cnn.encoding import encode_board


class AfterstateDataset(Dataset):
    """Compact afterstate-value dataset with optional random D4 augmentation."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        dataset: dict[str, np.ndarray | dict[str, Any]] | None = None,
        indices: np.ndarray | None = None,
        num_channels: int = 16,
        augment: bool = False,
        seed: int = 42,
    ) -> None:
        if dataset is None:
            if path is None:
                raise ValueError("path is required when dataset is not provided")
            dataset = load_expectimax_dataset(path)
        self.dataset = dataset
        self.boards = np.asarray(dataset["after_boards"], dtype=np.uint8)
        self.targets = np.asarray(dataset["target_us"], dtype=np.float32)
        self.root_ids = np.asarray(
            dataset.get("root_ids", np.arange(len(self.boards))),
            dtype=np.int64,
        )
        self.episodes = np.asarray(
            dataset.get("episodes", np.arange(len(self.boards))),
            dtype=np.int64,
        )
        self.indices = (
            np.arange(len(self.boards), dtype=np.int64)
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )
        self.num_channels = int(num_channels)
        self.augment = bool(augment)
        self.rng = np.random.default_rng(seed)

    def subset(
        self,
        indices: np.ndarray,
        *,
        augment: bool,
        seed: int,
    ) -> "AfterstateDataset":
        return AfterstateDataset(
            dataset=self.dataset,
            indices=indices,
            num_channels=self.num_channels,
            augment=augment,
            seed=seed,
        )

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        data_index = int(self.indices[index])
        board = self.boards[data_index]
        if self.augment:
            board = apply_symmetry(board, int(self.rng.integers(8)))
        encoded = encode_board(board, num_channels=self.num_channels)
        return (
            torch.from_numpy(encoded),
            torch.tensor(self.targets[data_index], dtype=torch.float32),
        )


def split_grouped_indices(
    root_ids: np.ndarray,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    groups = np.asarray(root_ids, dtype=np.int64)
    if len(groups) < 1:
        raise ValueError("dataset must contain at least one sample")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")

    unique_groups = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)
    if len(unique_groups) == 1 or validation_fraction == 0.0:
        validation_groups = np.asarray([], dtype=np.int64)
    else:
        validation_count = int(round(len(unique_groups) * validation_fraction))
        validation_count = min(max(validation_count, 1), len(unique_groups) - 1)
        validation_groups = unique_groups[:validation_count]
    validation_mask = np.isin(groups, validation_groups)
    all_indices = np.arange(len(groups), dtype=np.int64)
    return all_indices[~validation_mask], all_indices[validation_mask]
