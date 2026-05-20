import numpy as np

from gymnasium_2048.agents.expectimax import (
    generate_expectimax_dataset,
    save_expectimax_dataset,
)
from gymnasium_2048.agents.supervised_cnn import (
    SupervisedCNNPolicy,
    SupervisedTrainingConfig,
    train_supervised_cnn,
)
from gymnasium_2048.envs import TwentyFortyEightEnv


def test_supervised_policy_loads_and_predicts_legal_action(tmp_path):
    data_path = tmp_path / "teacher_data.npz"
    out_dir = tmp_path / "cnn"
    dataset = generate_expectimax_dataset(
        episodes=1,
        depth=1,
        seed=11,
        max_steps=4,
        progress=False,
    )
    save_expectimax_dataset(dataset, data_path)
    train_supervised_cnn(
        SupervisedTrainingConfig(
            data_path=str(data_path),
            out_dir=str(out_dir),
            epochs=1,
            batch_size=2,
            seed=11,
        )
    )

    policy = SupervisedCNNPolicy.load(out_dir / "best.pt")
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
