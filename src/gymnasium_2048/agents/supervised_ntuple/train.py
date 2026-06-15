from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from gymnasium_2048.agents.expectimax.data import load_expectimax_dataset
from gymnasium_2048.agents.supervised_cnn.data import split_grouped_indices
from gymnasium_2048.agents.supervised_ntuple.model import (
    SupervisedNTupleModel,
    resolve_patterns,
)


@dataclass(frozen=True)
class SupervisedNTupleTrainingConfig:
    data_path: str
    model_path: str
    pattern_set: str = "rows_cols"
    custom_patterns: Sequence[Sequence[Sequence[int]]] | None = None
    num_values: int = 16
    learning_rate: float = 0.05
    epochs: int = 10
    shuffle: bool = True
    weight_decay: float = 0.0
    target_normalization: bool = True
    validation_fraction: float = 0.2
    seed: int = 42


def _mean_squared_error(
    model: SupervisedNTupleModel,
    boards: np.ndarray,
    targets: np.ndarray,
    indices: np.ndarray,
) -> float:
    if len(indices) == 0:
        return 0.0
    errors = [
        model.evaluate_afterstate(boards[index]) - float(targets[index])
        for index in indices
    ]
    return float(np.mean(np.square(errors)))


def train_supervised_ntuple(
    config: SupervisedNTupleTrainingConfig,
) -> dict[str, Any]:
    dataset = load_expectimax_dataset(config.data_path)
    boards = np.asarray(dataset["after_boards"], dtype=np.uint8)
    targets = np.asarray(dataset["target_us"], dtype=np.float32)
    root_ids = np.asarray(
        dataset.get("root_ids", np.arange(len(boards))),
        dtype=np.int64,
    )
    train_indices, validation_indices = split_grouped_indices(
        root_ids,
        validation_fraction=config.validation_fraction,
        seed=config.seed,
    )
    train_targets = targets[train_indices]
    target_mean = (
        float(np.mean(train_targets)) if config.target_normalization else 0.0
    )
    target_std = (
        float(np.std(train_targets)) if config.target_normalization else 1.0
    )
    if target_std < 1.0e-6:
        target_std = 1.0

    model = SupervisedNTupleModel(
        resolve_patterns(config.pattern_set, config.custom_patterns),
        num_values=config.num_values,
        target_mean=target_mean,
        target_std=target_std,
    )
    rng = np.random.default_rng(config.seed)
    history: list[dict[str, float | int]] = []

    for epoch in range(1, config.epochs + 1):
        epoch_indices = train_indices.copy()
        if config.shuffle:
            rng.shuffle(epoch_indices)
        for index in epoch_indices:
            model.update(
                boards[index],
                target=float(targets[index]),
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        history.append(
            {
                "epoch": epoch,
                "train_loss": _mean_squared_error(
                    model,
                    boards,
                    targets,
                    train_indices,
                ),
                "validation_loss": _mean_squared_error(
                    model,
                    boards,
                    targets,
                    validation_indices,
                ),
            }
        )

    model.save(config.model_path)
    history_path = Path(config.model_path).with_suffix(".history.json")
    history_path.write_text(
        json.dumps(
            {
                "config": asdict(config),
                "target_mean": target_mean,
                "target_std": target_std,
                "history": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "model": model,
        "model_path": str(config.model_path),
        "history_path": str(history_path),
        "history": history,
        "target_mean": target_mean,
        "target_std": target_std,
    }
