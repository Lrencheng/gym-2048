import numpy as np
import pytest

from gymnasium_2048.agents.expectimax import all_symmetries, save_expectimax_dataset
from gymnasium_2048.agents.supervised_ntuple import (
    SupervisedNTupleModel,
    SupervisedNTuplePolicy,
    SupervisedNTupleTrainingConfig,
    resolve_patterns,
    train_supervised_ntuple,
    tuple_index,
)
from gymnasium_2048.envs import TwentyFortyEightEnv


def test_tuple_index_is_stable_base_encoding():
    board = np.array(
        [
            [1, 2, 3, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    pattern = ((0, 0), (0, 1), (0, 2))

    assert tuple_index(board, pattern, num_values=4) == 27
    assert tuple_index(board.copy(), pattern, num_values=4) == 27


def test_symmetric_prediction_is_invariant_after_update():
    model = SupervisedNTupleModel(
        patterns=resolve_patterns("rows"),
        num_values=16,
    )
    board = np.array(
        [
            [1, 2, 3, 4],
            [0, 1, 2, 3],
            [0, 0, 1, 2],
            [0, 0, 0, 1],
        ],
        dtype=np.uint8,
    )
    model.update(board, target=10.0, learning_rate=0.5)

    expected = model.evaluate_afterstate(board)
    for transformed in all_symmetries(board):
        assert model.evaluate_afterstate(transformed) == pytest.approx(expected)


def _write_toy_dataset(path):
    boards = []
    targets = []
    for index in range(20):
        board = np.zeros((4, 4), dtype=np.uint8)
        board[index % 4, (index // 4) % 4] = 1 + index % 4
        boards.append(board)
        targets.append(float(2 * board.sum() + np.count_nonzero(board)))
    dataset = {
        "after_boards": np.asarray(boards, dtype=np.uint8),
        "target_us": np.asarray(targets, dtype=np.float32),
        "immediate_rewards": np.zeros(20, dtype=np.float32),
        "actions": np.zeros(20, dtype=np.int64),
        "depths": np.zeros(20, dtype=np.int64),
        "root_ids": np.arange(20, dtype=np.int64),
        "episodes": np.arange(20, dtype=np.int64) // 5,
        "metadata": {"depth": 0, "num_samples": 20},
    }
    save_expectimax_dataset(dataset, path)


def test_supervised_ntuple_training_decreases_loss_and_roundtrips(tmp_path):
    data_path = tmp_path / "teacher.npz"
    model_path = tmp_path / "ntuple.npz"
    _write_toy_dataset(data_path)

    result = train_supervised_ntuple(
        SupervisedNTupleTrainingConfig(
            data_path=str(data_path),
            model_path=str(model_path),
            pattern_set="rows",
            learning_rate=0.4,
            epochs=8,
            validation_fraction=0.2,
            seed=5,
            target_normalization=True,
        )
    )

    losses = [row["train_loss"] for row in result["history"]]
    assert min(losses[1:]) < losses[0]
    assert model_path.exists()

    loaded = SupervisedNTupleModel.load(model_path)
    board = np.zeros((4, 4), dtype=np.uint8)
    board[0, 0] = 3
    assert loaded.evaluate_afterstate(board) == pytest.approx(
        result["model"].evaluate_afterstate(board)
    )


def test_supervised_ntuple_policy_uses_expectimax_action_selection(tmp_path):
    model = SupervisedNTupleModel(
        patterns=resolve_patterns("rows"),
        num_values=16,
    )
    model_path = tmp_path / "ntuple.npz"
    model.save(model_path)

    policy = SupervisedNTuplePolicy(model_path, depth=0)
    board = np.array(
        [
            [1, 1, 0, 0],
            [2, 0, 0, 0],
            [3, 0, 0, 0],
            [4, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    action = policy.predict(board)
    _afterstate, _reward, is_legal = TwentyFortyEightEnv.apply_action(
        board,
        action,
    )

    assert is_legal
