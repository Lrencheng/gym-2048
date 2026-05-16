import numpy as np

from gymnasium_2048.agents.heuristic import (
    HeuristicPolicy,
    corner_max,
    edge_bonus,
    empty_cells,
    evaluate_board,
    merge_potential,
    monotonicity,
    smoothness,
)
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
