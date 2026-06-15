import numpy as np

from gymnasium_2048.agents.expectimax import (
    apply_player_action,
    evaluate_root_actions,
    expand_chance_node,
    expectimax_afterstate_value,
    player_state_value,
)


def test_apply_player_action_returns_afterstate_and_raw_reward():
    board = np.array(
        [
            [2, 2, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )

    afterstate, reward, is_legal = apply_player_action(board, action=3)

    np.testing.assert_array_equal(
        afterstate,
        np.array(
            [
                [3, 0, 0, 0],
                [2, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.uint8,
        ),
    )
    assert reward == 12
    assert is_legal


def test_expand_chance_node_probabilities_sum_to_one():
    afterstate = np.ones((4, 4), dtype=np.uint8)
    afterstate[2, 3] = 0

    outcomes = expand_chance_node(afterstate)

    assert len(outcomes) == 2
    assert np.isclose(sum(outcome.probability for outcome in outcomes), 1.0)
    assert sorted(outcome.probability for outcome in outcomes) == [0.1, 0.9]
    assert {int(outcome.board[2, 3]) for outcome in outcomes} == {1, 2}


def test_depth_zero_afterstate_value_calls_evaluator_directly():
    board = np.arange(16, dtype=np.uint8).reshape(4, 4)
    seen = []

    value = expectimax_afterstate_value(
        board,
        depth=0,
        evaluator=lambda afterstate: seen.append(afterstate.copy()) or 17.5,
    )

    assert value == 17.5
    np.testing.assert_array_equal(seen[0], board)


def test_root_q_equals_immediate_reward_plus_afterstate_value():
    board = np.array(
        [
            [1, 1, 0, 0],
            [2, 0, 0, 0],
            [3, 0, 0, 0],
            [4, 0, 0, 0],
        ],
        dtype=np.uint8,
    )

    result = evaluate_root_actions(
        board,
        depth=0,
        evaluator=lambda afterstate: float(np.sum(afterstate)),
    )

    np.testing.assert_allclose(
        result.scores[result.legal_mask],
        (
            result.immediate_rewards[result.legal_mask]
            + result.afterstate_values[result.legal_mask]
        ),
    )


def test_terminal_player_state_has_zero_future_value():
    terminal = np.array(
        [
            [1, 2, 1, 2],
            [2, 1, 2, 1],
            [1, 2, 1, 2],
            [2, 1, 2, 1],
        ],
        dtype=np.uint8,
    )

    value = player_state_value(
        terminal,
        depth=1,
        evaluator=lambda _afterstate: 999.0,
    )

    assert value == 0.0
