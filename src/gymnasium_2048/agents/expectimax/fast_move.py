from __future__ import annotations

import itertools

import numpy as np

NUM_ACTIONS = 4
ROW_TABLE_MAX_EXPONENT = 16
ROW_TABLE_BASE = ROW_TABLE_MAX_EXPONENT + 1
ROW_TABLE_SIZE = ROW_TABLE_BASE**4

_MOVED_ROWS = np.zeros((ROW_TABLE_SIZE, 4), dtype=np.uint8)
_ROW_REWARDS = np.zeros(ROW_TABLE_SIZE, dtype=np.int32)
_ROW_CHANGED = np.zeros(ROW_TABLE_SIZE, dtype=bool)
_ROW_TABLE_READY = False


def _row_key(row: np.ndarray | tuple[int, int, int, int]) -> int:
    a, b, c, d = (int(value) for value in row)
    return (((a * ROW_TABLE_BASE) + b) * ROW_TABLE_BASE + c) * ROW_TABLE_BASE + d


def _move_row_left(row: tuple[int, int, int, int]) -> tuple[np.ndarray, int, bool]:
    non_zero = [value for value in row if value != 0]
    merged: list[int] = []
    reward = 0
    index = 0
    while index < len(non_zero):
        value = non_zero[index]
        if index + 1 < len(non_zero) and value == non_zero[index + 1]:
            merged_value = value + 1
            merged.append(merged_value)
            reward += 2**merged_value
            index += 2
        else:
            merged.append(value)
            index += 1

    moved = np.asarray([*merged, *([0] * (4 - len(merged)))], dtype=np.uint8)
    changed = tuple(int(value) for value in moved) != row
    return moved, reward, changed


def _ensure_row_table() -> None:
    global _ROW_TABLE_READY  # pylint: disable=global-statement
    if _ROW_TABLE_READY:
        return

    for row in itertools.product(range(ROW_TABLE_BASE), repeat=4):
        key = _row_key(row)
        moved, reward, changed = _move_row_left(row)
        _MOVED_ROWS[key] = moved
        _ROW_REWARDS[key] = reward
        _ROW_CHANGED[key] = changed

    _ROW_TABLE_READY = True


def _lookup_row_left(row: np.ndarray) -> tuple[np.ndarray, int, bool]:
    if int(np.max(row)) > ROW_TABLE_MAX_EXPONENT:
        return _move_row_left(tuple(int(value) for value in row))

    _ensure_row_table()
    key = _row_key(row)
    return _MOVED_ROWS[key].copy(), int(_ROW_REWARDS[key]), bool(_ROW_CHANGED[key])


def _move_rows_left(rows: np.ndarray) -> tuple[np.ndarray, int, bool]:
    moved = np.zeros_like(rows, dtype=np.uint8)
    total_reward = 0
    changed = False
    for row_index in range(rows.shape[0]):
        moved_row, reward, row_changed = _lookup_row_left(rows[row_index])
        moved[row_index] = moved_row
        total_reward += reward
        changed = changed or row_changed
    return moved, total_reward, changed


def fast_apply_action(board: np.ndarray, action: int) -> tuple[np.ndarray, int, bool]:
    """Apply a 2048 action using precomputed 4-cell row movement lookup."""
    state = np.asarray(board, dtype=np.uint8)
    if action == 3:  # left
        return _move_rows_left(state)
    if action == 1:  # right
        moved, reward, changed = _move_rows_left(np.fliplr(state))
        return np.fliplr(moved), reward, changed
    if action == 0:  # up
        moved, reward, changed = _move_rows_left(state.T)
        return moved.T, reward, changed
    if action == 2:  # down
        moved, reward, changed = _move_rows_left(np.flipud(state).T)
        return np.flipud(moved.T), reward, changed
    raise ValueError(f"invalid action: {action}")
