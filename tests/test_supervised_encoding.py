import numpy as np

from gymnasium_2048.agents.supervised_cnn import encode_board


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
