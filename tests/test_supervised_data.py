import numpy as np
import torch

from gymnasium_2048.agents.expectimax import all_symmetries
from gymnasium_2048.agents.supervised_cnn import (
    AfterstateDataset,
    split_grouped_indices,
)


def _dataset_payload():
    boards = np.array(
        [
            np.arange(16, dtype=np.uint8).reshape(4, 4),
            np.flipud(np.arange(16, dtype=np.uint8).reshape(4, 4)),
            np.eye(4, dtype=np.uint8),
            np.fliplr(np.eye(4, dtype=np.uint8)),
        ]
    )
    return {
        "after_boards": boards,
        "target_us": np.array([1.5, 1.5, 3.0, 3.0], dtype=np.float32),
        "root_ids": np.array([10, 10, 20, 20], dtype=np.int64),
        "episodes": np.array([0, 0, 1, 1], dtype=np.int64),
        "metadata": {},
    }


def test_afterstate_dataset_random_symmetry_preserves_target():
    payload = _dataset_payload()
    dataset = AfterstateDataset(
        dataset=payload,
        augment=True,
        seed=7,
        num_channels=16,
    )

    encoded, target = dataset[0]
    decoded = torch.argmax(encoded, dim=0).numpy().astype(np.uint8)

    assert encoded.shape == (16, 4, 4)
    assert encoded.dtype == torch.float32
    assert target.dtype == torch.float32
    assert float(target) == 1.5
    assert decoded.tobytes() in {
        board.tobytes() for board in all_symmetries(payload["after_boards"][0])
    }


def test_grouped_split_keeps_root_samples_together():
    payload = _dataset_payload()

    train_indices, validation_indices = split_grouped_indices(
        root_ids=payload["root_ids"],
        validation_fraction=0.5,
        seed=3,
    )

    train_roots = set(payload["root_ids"][train_indices].tolist())
    validation_roots = set(payload["root_ids"][validation_indices].tolist())
    assert train_roots.isdisjoint(validation_roots)
