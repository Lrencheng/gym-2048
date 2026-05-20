from __future__ import annotations

import numpy as np

from gymnasium_2048.agents.expectimax.fast_move import fast_apply_action

ACTION_NAMES = ("up", "right", "down", "left")
NUM_ACTIONS = 4


def board_from_observation(observation: np.ndarray) -> np.ndarray:
    """Return the exponent board used internally by the environment."""
    array = np.asarray(observation)
    if array.ndim == 2:
        return array.astype(np.uint8, copy=True)
    if array.ndim == 3:
        return np.argmax(array, axis=-1).astype(np.uint8, copy=False)
    raise ValueError(f"expected a 2-D board or 3-D one-hot observation, got {array.shape}")


def legal_action_mask(board: np.ndarray) -> np.ndarray:
    """Compute which actions change the board."""
    state = board_from_observation(board)
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    for action in range(NUM_ACTIONS):
        _next_board, _reward, is_legal = fast_apply_action(
            board=state,
            action=action,
        )
        mask[action] = is_legal
    return mask


def normalize_legal_mask(board: np.ndarray, legal_mask: np.ndarray | None = None) -> np.ndarray:
    """Combine an optional caller mask with the environment-derived legal mask."""
    computed_mask = legal_action_mask(board)
    if legal_mask is None:
        return computed_mask

    caller_mask = np.asarray(legal_mask, dtype=bool)
    if caller_mask.shape != (NUM_ACTIONS,):
        raise ValueError(f"legal_mask must have shape ({NUM_ACTIONS},), got {caller_mask.shape}")
    return computed_mask & caller_mask


def count_empty(board: np.ndarray) -> int:
    return int(np.count_nonzero(np.asarray(board) == 0))


def max_tile_exponent(board: np.ndarray) -> int:
    return int(np.max(np.asarray(board)))


def max_tile_value(board: np.ndarray) -> int:
    exponent = max_tile_exponent(board)
    return 0 if exponent == 0 else int(2**exponent)


def masked_softmax(
    scores: np.ndarray,
    legal_mask: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    """Convert action scores to probabilities over legal actions only."""
    scores = np.asarray(scores, dtype=np.float64)
    mask = np.asarray(legal_mask, dtype=bool)
    if scores.shape != (NUM_ACTIONS,) or mask.shape != (NUM_ACTIONS,):
        raise ValueError("scores and legal_mask must both have shape (4,)")
    if not np.any(mask):
        return np.zeros(NUM_ACTIONS, dtype=np.float64)

    temperature = max(float(temperature), 1e-6)
    safe_scores = np.where(mask, scores, -np.inf)
    safe_scores = np.where(np.isfinite(safe_scores), safe_scores, -np.inf)

    if not np.any(np.isfinite(safe_scores[mask])):
        probabilities = np.zeros(NUM_ACTIONS, dtype=np.float64)
        probabilities[mask] = 1.0 / float(np.count_nonzero(mask))
        return probabilities

    scaled = safe_scores / temperature
    scaled -= np.max(scaled[mask])
    exp_scores = np.zeros(NUM_ACTIONS, dtype=np.float64)
    exp_scores[mask] = np.exp(scaled[mask])
    total = float(np.sum(exp_scores))
    if total <= 0.0 or not np.isfinite(total):
        probabilities = np.zeros(NUM_ACTIONS, dtype=np.float64)
        probabilities[mask] = 1.0 / float(np.count_nonzero(mask))
        return probabilities
    return exp_scores / total
