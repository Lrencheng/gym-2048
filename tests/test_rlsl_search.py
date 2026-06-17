import numpy as np

from gymnasium_2048.agents.expectimax import evaluate_root_actions
from gymnasium_2048.agents.RLSL import (
    choose_search_improved_action,
    search_improved_afterstate_value,
)


class BatchOnlyEvaluator:
    def __init__(self):
        self.batch_sizes = []

    def evaluate_afterstates(self, afterstates):
        boards = np.asarray(afterstates, dtype=np.uint8)
        if boards.ndim == 2:
            boards = boards[None, :, :]
        self.batch_sizes.append(len(boards))
        return boards.reshape(len(boards), -1).sum(axis=1).astype(np.float64)

    def __call__(self, _afterstate):
        raise AssertionError("RLSL should batch depth=1 leaf evaluation")


def test_depth_one_search_target_returns_zero_when_spawned_states_are_terminal():
    afterstate = np.array(
        [
            [1, 2, 3, 4],
            [2, 3, 4, 1],
            [3, 4, 1, 4],
            [4, 1, 3, 0],
        ],
        dtype=np.uint8,
    )

    value = search_improved_afterstate_value(
        afterstate,
        evaluator=lambda _leaf_afterstate: 999.0,
    )

    assert value == 0.0


def test_selected_root_afterstate_target_excludes_current_action_reward():
    board = np.array(
        [
            [1, 1, 0, 0],
            [2, 0, 0, 0],
            [3, 0, 0, 0],
            [4, 0, 0, 0],
        ],
        dtype=np.uint8,
    )

    decision = choose_search_improved_action(
        board,
        evaluator=lambda afterstate: float(np.sum(afterstate)),
        search_depth=1,
    )

    assert decision.immediate_reward > 0.0
    assert decision.root_score == decision.immediate_reward + decision.target_value
    assert decision.target_value != decision.root_score


def test_search_decision_exposes_samples_for_all_legal_root_actions():
    board = np.array(
        [
            [1, 1, 0, 0],
            [2, 0, 0, 0],
            [3, 0, 0, 0],
            [4, 0, 0, 0],
        ],
        dtype=np.uint8,
    )

    decision = choose_search_improved_action(
        board,
        evaluator=lambda afterstate: float(np.sum(afterstate)),
        search_depth=1,
    )

    legal_actions = np.flatnonzero(decision.legal_mask).astype(int).tolist()
    assert [sample.action for sample in decision.action_samples] == legal_actions
    assert len(decision.action_samples) > 1
    for sample in decision.action_samples:
        assert sample.root_score == sample.immediate_reward + sample.target_value
        if sample.immediate_reward > 0.0:
            assert sample.target_value != sample.root_score


def test_choose_search_improved_action_batches_depth_one_leaf_values():
    board = np.array(
        [
            [1, 1, 0, 0],
            [2, 0, 0, 0],
            [3, 0, 0, 0],
            [4, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    evaluator = BatchOnlyEvaluator()

    decision = choose_search_improved_action(
        board,
        evaluator=evaluator,
        search_depth=1,
    )
    expected = evaluate_root_actions(
        board,
        depth=1,
        evaluator=lambda afterstate: float(np.sum(afterstate)),
    )

    assert evaluator.batch_sizes
    assert len(evaluator.batch_sizes) == 1
    assert evaluator.batch_sizes[0] > 4
    np.testing.assert_allclose(decision.scores, expected.scores)
    assert decision.action == int(np.argmax(expected.scores))
