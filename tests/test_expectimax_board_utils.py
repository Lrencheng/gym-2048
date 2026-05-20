import numpy as np

from gymnasium_2048.agents.expectimax import (
    board_from_observation,
    fast_apply_action,
    legal_action_mask,
    masked_softmax,
)
from gymnasium_2048.envs import TwentyFortyEightEnv


def test_board_from_one_hot_observation():
    observation = np.zeros((4, 4, 16), dtype=np.uint8)
    observation[:, :, 0] = 1
    observation[0, 0, 0] = 0
    observation[0, 0, 3] = 1

    board = board_from_observation(observation)

    assert board.shape == (4, 4)
    assert board[0, 0] == 3
    assert board[1, 1] == 0


def test_legal_action_mask_matches_board_motion():
    board = np.array(
        [
            [1, 1, 0, 0],
            [2, 0, 0, 0],
            [3, 0, 0, 0],
            [4, 0, 0, 0],
        ],
        dtype=np.uint8,
    )

    mask = legal_action_mask(board)

    assert mask.shape == (4,)
    assert mask.dtype == bool
    assert mask[1]
    assert mask[3]


def test_masked_softmax_ignores_illegal_actions():
    probabilities = masked_softmax(
        scores=np.array([1.0, 2.0, 100.0, -5.0]),
        legal_mask=np.array([True, True, False, False]),
    )

    assert np.isclose(probabilities.sum(), 1.0)
    assert probabilities[2] == 0.0
    assert probabilities[1] > probabilities[0]


def test_fast_apply_action_matches_environment_moves():
    board = np.array(
        [
            [2, 2, 2, 0],
            [1, 0, 1, 1],
            [2, 2, 0, 2],
            [0, 1, 1, 2],
        ],
        dtype=np.uint8,
    )

    for action in range(4):
        fast_board, fast_reward, fast_is_legal = fast_apply_action(board, action)
        env_board, env_reward, env_is_legal = TwentyFortyEightEnv.apply_action(
            board,
            action,
        )

        np.testing.assert_array_equal(fast_board, env_board)
        assert fast_reward == env_reward
        assert fast_is_legal == env_is_legal
