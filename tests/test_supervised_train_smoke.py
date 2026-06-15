import numpy as np
import pytest
import torch

from gymnasium_2048.agents.expectimax import save_expectimax_dataset
from gymnasium_2048.agents.supervised_cnn import (
    CNNAfterstateEvaluator,
    SupervisedTrainingConfig,
    encode_board,
    resolve_device,
    train_supervised_cnn,
)


def _write_toy_dataset(path):
    boards = []
    targets = []
    root_ids = []
    episodes = []
    for index in range(24):
        board = np.zeros((4, 4), dtype=np.uint8)
        board[index % 4, (index // 4) % 4] = 1 + index % 5
        boards.append(board)
        targets.append(float(board.sum() * 2.0 + np.count_nonzero(board == 0)))
        root_ids.append(index)
        episodes.append(index // 6)
    dataset = {
        "after_boards": np.asarray(boards, dtype=np.uint8),
        "target_us": np.asarray(targets, dtype=np.float32),
        "immediate_rewards": np.zeros(24, dtype=np.float32),
        "actions": np.zeros(24, dtype=np.int64),
        "depths": np.zeros(24, dtype=np.int64),
        "root_ids": np.asarray(root_ids, dtype=np.int64),
        "episodes": np.asarray(episodes, dtype=np.int64),
        "metadata": {"depth": 0, "num_samples": 24},
    }
    save_expectimax_dataset(dataset, path)


def test_supervised_train_toy_loss_decreases_and_checkpoint_loads(tmp_path):
    data_path = tmp_path / "teacher_data.npz"
    out_dir = tmp_path / "cnn"
    _write_toy_dataset(data_path)

    result = train_supervised_cnn(
        SupervisedTrainingConfig(
            data_path=str(data_path),
            out_dir=str(out_dir),
            epochs=8,
            batch_size=6,
            learning_rate=0.01,
            validation_fraction=0.25,
            seed=9,
            device="cpu",
            num_workers=0,
            target_normalization=True,
            symmetry_augmentation=True,
        )
    )

    losses = [row["train_loss"] for row in result["history"]]
    assert min(losses[1:]) < losses[0]
    assert (out_dir / "checkpoints" / "best.pt").exists()
    assert (out_dir / "checkpoints" / "last.pt").exists()
    assert (out_dir / "history.json").exists()

    board = np.zeros((4, 4), dtype=np.uint8)
    encoded = torch.from_numpy(encode_board(board)).unsqueeze(0)
    result["model"].eval()
    with torch.no_grad():
        normalized = float(result["model"](encoded).item())
    expected = normalized * result["target_std"] + result["target_mean"]
    evaluator = CNNAfterstateEvaluator.load(result["last_checkpoint"])
    assert evaluator.evaluate_afterstate(board) == pytest.approx(expected)


def test_resolve_device_auto_returns_valid_torch_device():
    device = resolve_device("auto")

    assert device.type in {"cpu", "cuda"}
