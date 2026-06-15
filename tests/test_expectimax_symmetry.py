import numpy as np

from gymnasium_2048.agents.expectimax import (
    all_symmetries,
    apply_player_action,
    apply_symmetry,
    evaluate_board,
    expectimax_afterstate_value,
    random_symmetry,
    transform_action,
)


def test_all_symmetries_returns_eight_transforms():
    board = np.arange(16, dtype=np.uint8).reshape(4, 4)

    transformed = all_symmetries(board)

    assert transformed.shape == (8, 4, 4)
    assert len({item.tobytes() for item in transformed}) == 8


def test_transformed_action_commutes_with_board_transform():
    board = np.array(
        [
            [1, 1, 0, 0],
            [2, 0, 2, 0],
            [3, 0, 0, 3],
            [4, 4, 0, 0],
        ],
        dtype=np.uint8,
    )

    for sym_id in range(8):
        transformed_board = apply_symmetry(board, sym_id)
        for action in range(4):
            afterstate, reward, is_legal = apply_player_action(board, action)
            mapped_action = transform_action(action, sym_id)
            mapped_afterstate, mapped_reward, mapped_is_legal = apply_player_action(
                transformed_board,
                mapped_action,
            )

            np.testing.assert_array_equal(
                apply_symmetry(afterstate, sym_id),
                mapped_afterstate,
            )
            assert mapped_reward == reward
            assert mapped_is_legal == is_legal


def test_random_symmetry_returns_one_of_the_eight_transforms():
    board = np.arange(16, dtype=np.uint8).reshape(4, 4)
    expected = {item.tobytes() for item in all_symmetries(board)}

    transformed = random_symmetry(board, rng=np.random.default_rng(7))

    assert transformed.tobytes() in expected


def test_symmetric_heuristic_gives_symmetric_depth_zero_value():
    board = np.array(
        [
            [8, 7, 6, 5],
            [1, 2, 3, 4],
            [0, 1, 2, 3],
            [0, 0, 1, 2],
        ],
        dtype=np.uint8,
    )

    expected = expectimax_afterstate_value(board, depth=0, evaluator=evaluate_board)

    for transformed in all_symmetries(board):
        actual = expectimax_afterstate_value(
            transformed,
            depth=0,
            evaluator=evaluate_board,
        )
        assert actual == expected


def test_exact_depth_one_search_is_symmetric():
    board = np.array(
        [
            [8, 7, 6, 5],
            [1, 2, 3, 4],
            [4, 3, 2, 1],
            [0, 0, 1, 2],
        ],
        dtype=np.uint8,
    )
    expected = expectimax_afterstate_value(board, depth=1, evaluator=evaluate_board)

    for transformed in all_symmetries(board):
        actual = expectimax_afterstate_value(
            transformed,
            depth=1,
            evaluator=evaluate_board,
        )
        assert actual == expected
