from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from gymnasium_2048.agents.expectimax.symmetry import all_symmetries

Coordinate = tuple[int, int]
Pattern = tuple[Coordinate, ...]


def resolve_patterns(
    pattern_set: str,
    custom_patterns: Sequence[Sequence[Sequence[int]]] | None = None,
) -> tuple[Pattern, ...]:
    if pattern_set == "custom":
        if not custom_patterns:
            raise ValueError("custom_patterns are required for pattern_set='custom'")
        return tuple(
            tuple((int(row), int(col)) for row, col in pattern)
            for pattern in custom_patterns
        )

    rows = tuple(
        tuple((row, col) for col in range(4))
        for row in range(4)
    )
    cols = tuple(
        tuple((row, col) for row in range(4))
        for col in range(4)
    )
    if pattern_set == "rows":
        return rows
    if pattern_set == "cols":
        return cols
    if pattern_set == "rows_cols":
        return rows + cols
    if pattern_set == "rectangles_2x3":
        patterns: list[Pattern] = []
        for row in range(3):
            for col in range(2):
                patterns.append(
                    tuple(
                        (row + row_offset, col + col_offset)
                        for row_offset in range(2)
                        for col_offset in range(3)
                    )
                )
        for row in range(2):
            for col in range(3):
                patterns.append(
                    tuple(
                        (row + row_offset, col + col_offset)
                        for row_offset in range(3)
                        for col_offset in range(2)
                    )
                )
        return tuple(patterns)
    if pattern_set == "snake":
        return (
            ((0, 0), (0, 1), (0, 2), (0, 3), (1, 3), (1, 2)),
            ((1, 0), (1, 1), (1, 2), (1, 3), (2, 3), (2, 2)),
            ((2, 0), (2, 1), (2, 2), (2, 3), (3, 3), (3, 2)),
            ((3, 0), (3, 1), (3, 2), (3, 3), (2, 3), (2, 2)),
        )
    raise ValueError(f"unknown pattern_set: {pattern_set!r}")


def tuple_index(
    board: np.ndarray,
    pattern: Pattern,
    num_values: int,
) -> int:
    if num_values < 2:
        raise ValueError("num_values must be at least two")
    state = np.asarray(board)
    index = 0
    for row, col in pattern:
        value = int(np.clip(state[row, col], 0, num_values - 1))
        index = index * num_values + value
    return index


class SupervisedNTupleModel:
    """Sparse symmetric n-tuple regressor for afterstate continuation value."""

    def __init__(
        self,
        patterns: Sequence[Pattern],
        *,
        num_values: int = 16,
        target_mean: float = 0.0,
        target_std: float = 1.0,
    ) -> None:
        if not patterns:
            raise ValueError("at least one tuple pattern is required")
        self.patterns = tuple(tuple(pattern) for pattern in patterns)
        self.num_values = int(num_values)
        self.target_mean = float(target_mean)
        self.target_std = max(float(target_std), 1.0e-6)
        self.tables: list[dict[int, float]] = [
            {} for _pattern in self.patterns
        ]

    def _feature_counts(
        self,
        board: np.ndarray,
    ) -> list[dict[int, int]]:
        counts: list[dict[int, int]] = [
            {} for _pattern in self.patterns
        ]
        for transformed in all_symmetries(board):
            for pattern_index, pattern in enumerate(self.patterns):
                key = tuple_index(transformed, pattern, self.num_values)
                counts[pattern_index][key] = (
                    counts[pattern_index].get(key, 0) + 1
                )
        return counts

    def predict_normalized(self, board: np.ndarray) -> float:
        total = 0.0
        for transformed in all_symmetries(board):
            total += sum(
                self.tables[pattern_index].get(
                    tuple_index(transformed, pattern, self.num_values),
                    0.0,
                )
                for pattern_index, pattern in enumerate(self.patterns)
            )
        return total / 8.0

    def evaluate_afterstate(self, after_board: np.ndarray) -> float:
        normalized = self.predict_normalized(after_board)
        return normalized * self.target_std + self.target_mean

    def __call__(self, after_board: np.ndarray) -> float:
        return self.evaluate_afterstate(after_board)

    def update(
        self,
        board: np.ndarray,
        *,
        target: float,
        learning_rate: float,
        weight_decay: float = 0.0,
    ) -> float:
        normalized_target = (float(target) - self.target_mean) / self.target_std
        prediction = self.predict_normalized(board)
        residual = normalized_target - prediction
        counts = self._feature_counts(board)
        gradients = [
            {
                key: count / 8.0
                for key, count in table_counts.items()
            }
            for table_counts in counts
        ]
        gradient_norm = sum(
            gradient * gradient
            for table_gradients in gradients
            for gradient in table_gradients.values()
        )
        if gradient_norm <= 0.0:
            return residual * residual

        for table, table_gradients in zip(self.tables, gradients):
            for key, gradient in table_gradients.items():
                old_value = table.get(key, 0.0)
                if weight_decay:
                    old_value *= max(0.0, 1.0 - learning_rate * weight_decay)
                table[key] = (
                    old_value
                    + learning_rate * residual * gradient / gradient_norm
                )
        return residual * residual

    def save(self, path: str | Path) -> None:
        arrays: dict[str, np.ndarray] = {}
        for index, table in enumerate(self.tables):
            keys = np.asarray(sorted(table), dtype=np.int64)
            values = np.asarray([table[int(key)] for key in keys], dtype=np.float32)
            arrays[f"keys_{index}"] = keys
            arrays[f"values_{index}"] = values
        metadata = {
            "patterns": [
                [[row, col] for row, col in pattern]
                for pattern in self.patterns
            ],
            "num_values": self.num_values,
            "target_mean": self.target_mean,
            "target_std": self.target_std,
            "symmetry_aggregation": "mean_8",
        }
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            **arrays,
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "SupervisedNTupleModel":
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
            patterns = resolve_patterns(
                "custom",
                custom_patterns=metadata["patterns"],
            )
            model = cls(
                patterns,
                num_values=int(metadata["num_values"]),
                target_mean=float(metadata.get("target_mean", 0.0)),
                target_std=float(metadata.get("target_std", 1.0)),
            )
            for index, table in enumerate(model.tables):
                keys = data[f"keys_{index}"]
                values = data[f"values_{index}"]
                table.update(
                    {
                        int(key): float(value)
                        for key, value in zip(keys, values)
                    }
                )
            return model
