import json

import numpy as np

from gymnasium_2048.agents.SL1 import (
    SupervisedCNNPolicy,
    SupervisedTrainingConfig,
    train_supervised_cnn,
)
from gymnasium_2048.envs import TwentyFortyEightEnv


def test_sl1_policy_loads_and_predicts_legal_action(tmp_path):
    data_path = tmp_path / "teacher_data.npz"
    out_dir = tmp_path / "cnn"
    boards = np.asarray(
        [
            [[1, 1, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0]],
            [[0, 0, 1, 1], [0, 0, 0, 2], [0, 0, 0, 3], [0, 0, 0, 4]],
            [[1, 0, 0, 0], [1, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4]],
            [[0, 0, 0, 1], [0, 0, 2, 1], [0, 3, 0, 0], [4, 0, 0, 0]],
        ],
        dtype=np.uint8,
    )
    np.savez_compressed(
        data_path,
        boards=boards,
        legal_masks=np.asarray(
            [
                [1, 1, 1, 0],
                [1, 0, 1, 1],
                [1, 1, 0, 1],
                [0, 1, 1, 1],
            ],
            dtype=bool,
        ),
        action_probs=np.asarray(
            [
                [0.80, 0.10, 0.10, 0.00],
                [0.70, 0.00, 0.20, 0.10],
                [0.60, 0.30, 0.00, 0.10],
                [0.00, 0.60, 0.30, 0.10],
            ],
            dtype=np.float32,
        ),
        actions=np.asarray([0, 0, 0, 1], dtype=np.int64),
        final_scores=np.asarray([1024, 2048, 4096, 8192], dtype=np.int64),
        final_max_tiles=np.asarray([128, 256, 512, 1024], dtype=np.int64),
        empty_counts=np.asarray([10, 9, 8, 7], dtype=np.int64),
        steps=np.asarray([1, 2, 3, 4], dtype=np.int64),
        metadata=np.asarray(json.dumps({"target": "action_probabilities"})),
    )
    result = train_supervised_cnn(
        SupervisedTrainingConfig(
            data_path=str(data_path),
            out_dir=str(out_dir),
            epochs=1,
            batch_size=2,
            seed=11,
            device="cpu",
            num_workers=0,
            amp=False,
        )
    )

    policy = SupervisedCNNPolicy.load(result["best_checkpoint"])
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
    _next_board, _reward, is_legal = TwentyFortyEightEnv.apply_action(board, action)

    assert is_legal
