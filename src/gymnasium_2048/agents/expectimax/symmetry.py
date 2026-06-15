from __future__ import annotations

import numpy as np

from gymnasium_2048.agents.expectimax.fast_move import fast_apply_action

NUM_SYMMETRIES = 8


def apply_symmetry(board: np.ndarray, sym_id: int) -> np.ndarray:
    """Apply one of the eight D4 symmetries to a board."""
    if not 0 <= int(sym_id) < NUM_SYMMETRIES:
        raise ValueError(f"sym_id must be in [0, {NUM_SYMMETRIES}), got {sym_id}")

    state = np.asarray(board)
    rotations = int(sym_id) % 4
    transformed = np.rot90(state, rotations)
    if int(sym_id) >= 4:
        transformed = np.fliplr(transformed)
    return np.ascontiguousarray(transformed)


def all_symmetries(board: np.ndarray) -> np.ndarray:
    """Return all eight D4 transforms in symmetry-ID order."""
    return np.stack(
        [apply_symmetry(board, sym_id) for sym_id in range(NUM_SYMMETRIES)]
    )


def random_symmetry(
    board: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return one uniformly sampled D4 transform."""
    generator = rng or np.random.default_rng()
    sym_id = int(generator.integers(NUM_SYMMETRIES))
    return apply_symmetry(board, sym_id)


def _build_action_transforms() -> np.ndarray:
    probe = np.zeros((4, 4), dtype=np.uint8)
    probe[1, 1] = 1
    transforms = np.zeros((NUM_SYMMETRIES, 4), dtype=np.int64)

    for sym_id in range(NUM_SYMMETRIES):
        transformed_probe = apply_symmetry(probe, sym_id)
        for action in range(4):
            afterstate, _reward, _legal = fast_apply_action(probe, action)
            expected = apply_symmetry(afterstate, sym_id)
            for candidate in range(4):
                actual, _candidate_reward, _candidate_legal = fast_apply_action(
                    transformed_probe,
                    candidate,
                )
                if np.array_equal(actual, expected):
                    transforms[sym_id, action] = candidate
                    break
            else:
                raise RuntimeError(
                    f"could not map action {action} through symmetry {sym_id}"
                )
    return transforms


_ACTION_TRANSFORMS = _build_action_transforms()


def transform_action(action: int, sym_id: int) -> int:
    """Transform an action direction with the same symmetry as its board."""
    if not 0 <= int(action) < 4:
        raise ValueError(f"action must be in [0, 4), got {action}")
    if not 0 <= int(sym_id) < NUM_SYMMETRIES:
        raise ValueError(f"sym_id must be in [0, {NUM_SYMMETRIES}), got {sym_id}")
    return int(_ACTION_TRANSFORMS[int(sym_id), int(action)])
