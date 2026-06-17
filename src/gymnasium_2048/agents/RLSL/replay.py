from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from gymnasium_2048.agents.expectimax.data import load_expectimax_dataset
from gymnasium_2048.agents.expectimax.symmetry import apply_symmetry
from gymnasium_2048.agents.supervised_cnn.encoding import encode_board

REPLAY_SOURCE_TEACHER = 0
REPLAY_SOURCE_ONLINE = 1


class RLSLReplayBuffer:
    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("replay capacity must be positive")
        self.capacity = int(capacity)
        self.boards = np.empty((self.capacity, 4, 4), dtype=np.uint8)
        self.targets = np.empty((self.capacity,), dtype=np.float32)
        self.sources = np.empty((self.capacity,), dtype=np.uint8)
        self.start = 0
        self.size = 0

    @classmethod
    def from_arrays(
        cls,
        boards: np.ndarray,
        targets: np.ndarray,
        *,
        capacity: int,
        seed: int,
        shuffle: bool = True,
    ) -> "RLSLReplayBuffer":
        buffer = cls(capacity)
        boards_array = np.asarray(boards, dtype=np.uint8)
        targets_array = np.asarray(targets, dtype=np.float32).reshape(-1)
        if len(boards_array) != len(targets_array):
            raise ValueError("teacher boards and targets must have the same length")

        sample_count = min(len(boards_array), buffer.capacity)
        if sample_count == 0:
            return buffer
        if shuffle:
            rng = np.random.default_rng(seed)
            indices = rng.choice(len(boards_array), size=sample_count, replace=False)
        else:
            indices = np.arange(sample_count, dtype=np.int64)
        buffer._append_arrays(
            boards_array[indices],
            targets_array[indices],
            np.full(sample_count, REPLAY_SOURCE_TEACHER, dtype=np.uint8),
        )
        return buffer

    @classmethod
    def from_teacher_dataset(
        cls,
        path: str | Path,
        *,
        capacity: int,
        seed: int,
    ) -> "RLSLReplayBuffer":
        dataset = load_expectimax_dataset(path)
        return cls.from_arrays(
            np.asarray(dataset["after_boards"], dtype=np.uint8),
            np.asarray(dataset["target_us"], dtype=np.float32),
            capacity=capacity,
            seed=seed,
        )

    def __len__(self) -> int:
        return int(self.size)

    def _physical_index(self, logical_index: int) -> int:
        if logical_index < 0 or logical_index >= self.size:
            raise IndexError(logical_index)
        return (self.start + logical_index) % self.capacity

    def _append_arrays(
        self,
        boards: np.ndarray,
        targets: np.ndarray,
        sources: np.ndarray,
    ) -> None:
        for board, target, source in zip(boards, targets, sources):
            if self.size < self.capacity:
                write_index = (self.start + self.size) % self.capacity
                self.size += 1
            else:
                write_index = self.start
                self.start = (self.start + 1) % self.capacity
            self.boards[write_index] = np.asarray(board, dtype=np.uint8)
            self.targets[write_index] = np.float32(target)
            self.sources[write_index] = np.uint8(source)

    def append_online(self, samples: list[Any]) -> None:
        if not samples:
            return
        boards = np.asarray([sample.afterstate for sample in samples], dtype=np.uint8)
        targets = np.asarray([sample.target_value for sample in samples], dtype=np.float32)
        sources = np.full(len(samples), REPLAY_SOURCE_ONLINE, dtype=np.uint8)
        self._append_arrays(boards, targets, sources)

    def get(self, index: int) -> tuple[np.ndarray, float, int]:
        physical_index = self._physical_index(index)
        return (
            self.boards[physical_index],
            float(self.targets[physical_index]),
            int(self.sources[physical_index]),
        )

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        indices = [self._physical_index(index) for index in range(self.size)]
        return (
            self.boards[indices].copy(),
            self.targets[indices].copy(),
            self.sources[indices].copy(),
        )

    def source_counts(self) -> dict[int, int]:
        if self.size == 0:
            return {REPLAY_SOURCE_TEACHER: 0, REPLAY_SOURCE_ONLINE: 0}
        _boards, _targets, sources = self.arrays()
        return {
            REPLAY_SOURCE_TEACHER: int(np.sum(sources == REPLAY_SOURCE_TEACHER)),
            REPLAY_SOURCE_ONLINE: int(np.sum(sources == REPLAY_SOURCE_ONLINE)),
        }


def cap_current_samples(
    samples: list[Any],
    *,
    max_samples: int,
    rng: np.random.Generator,
) -> list[Any]:
    limit = int(max_samples)
    if limit < 1:
        return []
    if len(samples) <= limit:
        return list(samples)
    indices = rng.choice(len(samples), size=limit, replace=False)
    return [samples[int(index)] for index in indices]


def choose_admitted_samples(
    samples: list[Any],
    *,
    fraction: float,
    rng: np.random.Generator,
) -> list[Any]:
    if not 0.0 <= float(fraction) <= 1.0:
        raise ValueError("replay admission fraction must be in [0, 1]")
    if not samples:
        return []
    count = int(round(len(samples) * float(fraction)))
    count = min(max(count, 0), len(samples))
    if count == 0:
        return []
    if count == len(samples):
        return list(samples)
    indices = rng.choice(len(samples), size=count, replace=False)
    return [samples[int(index)] for index in indices]


class ReplayAfterstateDataset(Dataset):
    def __init__(
        self,
        *,
        buffer: RLSLReplayBuffer,
        current_samples: list[Any],
        config: Any,
        seed: int,
    ) -> None:
        self.buffer = buffer
        self.current_samples = list(current_samples)
        self.num_channels = int(config.input_channels)
        self.augment = bool(config.symmetry_augmentation)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.buffer) + len(self.current_samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        replay_size = len(self.buffer)
        if index < replay_size:
            board, target, _source = self.buffer.get(index)
        else:
            sample = self.current_samples[index - replay_size]
            board = np.asarray(sample.afterstate, dtype=np.uint8)
            target = float(sample.target_value)

        if self.augment:
            board = apply_symmetry(board, int(self.rng.integers(8)))
        encoded = encode_board(board, num_channels=self.num_channels)
        return (
            torch.from_numpy(encoded),
            torch.tensor(float(target), dtype=torch.float32),
        )
