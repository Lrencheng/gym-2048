import numpy as np

from gymnasium_2048.agents.expectimax import ExpectimaxPolicy
from gymnasium_2048.envs import TwentyFortyEightEnv


def test_expectimax_single_step_returns_legal_action_and_distribution():
    board = np.array(
        [
            [1, 1, 0, 0],
            [2, 0, 0, 0],
            [3, 0, 0, 0],
            [4, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    policy = ExpectimaxPolicy(depth=1, seed=7)

    result = policy.analyze(board)
    _next_board, _reward, is_legal = TwentyFortyEightEnv.apply_action(
        board,
        result.action,
    )

    assert is_legal
    assert result.scores.shape == (4,)
    assert result.legal_mask.shape == (4,)
    assert np.isclose(result.probabilities.sum(), 1.0)
    assert result.probabilities[~result.legal_mask].sum() == 0.0


def test_expectimax_adaptive_chance_sampling_threshold():
    empty_cells = np.argwhere(np.zeros((4, 4), dtype=np.uint8) == 0)
    policy = ExpectimaxPolicy(
        depth=2,
        seed=7,
        chance_samples=6,
        full_chance_empty_threshold=6,
    )

    sampled = policy._sample_empty_cells(empty_cells)
    exact_small = policy._sample_empty_cells(empty_cells[:6])

    assert sampled.shape == (6, 2)
    assert exact_small.shape == (6, 2)
    assert np.array_equal(exact_small, empty_cells[:6])
