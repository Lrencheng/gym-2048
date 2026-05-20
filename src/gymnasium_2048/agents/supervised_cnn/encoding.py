from __future__ import annotations

import numpy as np

from gymnasium_2048.agents.expectimax.board import board_from_observation


def encode_board(board: np.ndarray, num_channels: int = 16) -> np.ndarray:
    """One-hot encode an exponent board as C x 4 x 4 float32 input."""
    if num_channels < 2:
        raise ValueError("num_channels must be at least 2")

    state = board_from_observation(board)
    clipped = np.clip(state, 0, num_channels - 1).astype(np.int64, copy=False)
    encoded = np.zeros((num_channels, *clipped.shape), dtype=np.float32)
    rows, cols = np.indices(clipped.shape)
    encoded[clipped, rows, cols] = 1.0
    return encoded


def encode_boards(boards: np.ndarray, num_channels: int = 16) -> np.ndarray:
    states = np.asarray(boards)
    if states.ndim != 3:
        raise ValueError(f"expected boards with shape (N, H, W), got {states.shape}")
    return np.stack([encode_board(board, num_channels=num_channels) for board in states])
