import numpy as np

from gymnasium_2048.agents.expectimax import save_expectimax_dataset
from gymnasium_2048.agents.supervised_cnn import (
    CNNAfterstateEvaluator,
    SupervisedCNNPolicy,
    SupervisedTrainingConfig,
    train_supervised_cnn,
)
from gymnasium_2048.envs import TwentyFortyEightEnv


def test_cnn_evaluator_is_stable_after_load_and_policy_is_legal(tmp_path):
    data_path = tmp_path / "teacher_data.npz"
    out_dir = tmp_path / "cnn"
    boards = np.zeros((8, 4, 4), dtype=np.uint8)
    for index in range(8):
        boards[index, index % 4, index // 4] = 1 + index % 3
    dataset = {
        "after_boards": boards,
        "target_us": np.arange(8, dtype=np.float32),
        "immediate_rewards": np.zeros(8, dtype=np.float32),
        "actions": np.zeros(8, dtype=np.int64),
        "depths": np.zeros(8, dtype=np.int64),
        "root_ids": np.arange(8, dtype=np.int64),
        "episodes": np.arange(8, dtype=np.int64),
        "metadata": {"depth": 0, "num_samples": 8},
    }
    save_expectimax_dataset(dataset, data_path)
    result = train_supervised_cnn(
        SupervisedTrainingConfig(
            data_path=str(data_path),
            out_dir=str(out_dir),
            epochs=1,
            batch_size=4,
            validation_fraction=0.25,
            seed=11,
            device="cpu",
            num_workers=0,
        )
    )

    evaluator_a = CNNAfterstateEvaluator.load(result["best_checkpoint"])
    evaluator_b = CNNAfterstateEvaluator.load(result["best_checkpoint"])
    board = np.array(
        [
            [1, 1, 0, 0],
            [2, 0, 0, 0],
            [3, 0, 0, 0],
            [4, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    assert evaluator_a(board) == evaluator_b(board)

    policy = SupervisedCNNPolicy(
        checkpoint=result["best_checkpoint"],
        depth=0,
    )
    action = policy.predict(board)
    _afterstate, _reward, is_legal = TwentyFortyEightEnv.apply_action(
        board,
        action,
    )
    assert is_legal
