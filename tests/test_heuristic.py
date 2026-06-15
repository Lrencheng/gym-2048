import numpy as np

from gymnasium_2048.agents.heuristic import (
    HeuristicPolicy,
    corner_max,
    edge_bonus,
    empty_cells,
    evaluate_board,
    merge_potential,
    monotonicity,
    snake_score,
    smoothness,
    transform_reward,
)
from gymnasium_2048.agents.heuristic.features import SNAKE_WEIGHTS
from gymnasium_2048.envs import TwentyFortyEightEnv


def test_heuristic_features():
    board = np.array(
        [
            [4, 3, 2, 1],
            [0, 0, 2, 1],
            [3, 3, 0, 1],
            [0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )

    assert empty_cells(board) == 7
    assert smoothness(board) == -4
    assert monotonicity(board) == 0
    assert corner_max(board) == 4
    assert merge_potential(board) == 4
    assert edge_bonus(board) == 42
    assert evaluate_board(board, reward=8) > 0


def test_reward_transform_modes():
    assert transform_reward(8, "raw") == 8
    assert np.isclose(transform_reward(8, "log2p1"), np.log2(9))
    assert transform_reward(8, "none") == 0


def test_snake_score_is_invariant_under_all_board_symmetries():
    board = np.array(
        [
            [1, 0, 2, 4],
            [3, 5, 0, 1],
            [0, 2, 6, 3],
            [7, 1, 4, 0],
        ],
        dtype=np.uint8,
    )
    symmetries = [
        transformed
        for rotations in range(4)
        for transformed in (
            np.rot90(board, rotations),
            np.fliplr(np.rot90(board, rotations)),
        )
    ]
    expected = max(
        float(np.sum(transformed * SNAKE_WEIGHTS) / 16.0)
        for transformed in symmetries
    )

    scores = [snake_score(transformed) for transformed in symmetries]

    assert all(np.isclose(score, expected) for score in scores)


def test_heuristic_policy_predicts_legal_best_action():
    policy = HeuristicPolicy()
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
    scores = [policy.evaluate(board, candidate) for candidate in range(4)]
    _, _, is_legal = TwentyFortyEightEnv.apply_action(board, action)

    assert is_legal
    assert scores[action] == max(scores)


def test_heuristic_policy_returns_zero_when_no_action_is_legal():
    policy = HeuristicPolicy()
    board = np.array(
        [
            [1, 2, 1, 2],
            [2, 1, 2, 1],
            [1, 2, 1, 2],
            [2, 1, 2, 1],
        ],
        dtype=np.uint8,
    )

    assert policy.predict(board) == 0
