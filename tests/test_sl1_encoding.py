import numpy as np
import torch

from gymnasium_2048.agents.SL1 import encode_board, encode_boards_torch


def test_encode_board_one_hot_and_clamps_large_tiles():
    board = np.array(
        [
            [0, 1, 2, 20],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )

    encoded = encode_board(board, num_channels=16)

    assert encoded.shape == (16, 4, 4)
    assert encoded[:, 0, 0].sum() == 1.0
    assert encoded[0, 0, 0] == 1.0
    assert encoded[1, 0, 1] == 1.0
    assert encoded[15, 0, 3] == 1.0


def test_torch_batch_encoding_matches_numpy_encoding():
    boards = np.array(
        [
            [
                [0, 1, 2, 20],
                [3, 4, 5, 6],
                [7, 8, 9, 10],
                [11, 12, 13, 14],
            ],
            [
                [1, 0, 0, 0],
                [0, 2, 0, 0],
                [0, 0, 3, 0],
                [0, 0, 0, 4],
            ],
        ],
        dtype=np.uint8,
    )

    expected = np.stack([encode_board(board, num_channels=16) for board in boards])
    actual = encode_boards_torch(torch.from_numpy(boards), num_channels=16)

    np.testing.assert_array_equal(actual.numpy(), expected)

