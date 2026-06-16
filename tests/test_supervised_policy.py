import numpy as np

from gymnasium_2048.agents.expectimax import (
    evaluate_root_actions,
    save_expectimax_dataset,
)
from gymnasium_2048.agents.supervised_cnn import (
    CNNAfterstateEvaluator,
    SupervisedCNNPolicy,
    SupervisedTrainingConfig,
    train_supervised_cnn,
)
from gymnasium_2048.agents.supervised_cnn.policy import (
    _evaluate_root_actions_batched,
)
from gymnasium_2048.envs import TwentyFortyEightEnv


def test_cnn_evaluator_is_stable_after_load_and_policy_is_legal(tmp_path):
    data_path = tmp_path / "teacher_data.npz"
    out_dir = tmp_path / "cnn"
    boards = np.zeros((8, 4, 4), dtype=np.uint8)
    for index in range(8):
        boards[index, index % 4, index // 4] = 1 + index % 3
    dataset = {
        "after_boards": boards,
        "target_us": np.arange(8, dtype=np.float32),
        "immediate_rewards": np.zeros(8, dtype=np.float32),
        "actions": np.zeros(8, dtype=np.int64),
        "depths": np.zeros(8, dtype=np.int64),
        "root_ids": np.arange(8, dtype=np.int64),
        "episodes": np.arange(8, dtype=np.int64),
        "metadata": {"depth": 0, "num_samples": 8},
    }
    save_expectimax_dataset(dataset, data_path)
    result = train_supervised_cnn(
        SupervisedTrainingConfig(
            data_path=str(data_path),
            out_dir=str(out_dir),
            epochs=1,
            batch_size=4,
            validation_fraction=0.25,
            seed=11,
            device="cpu",
            num_workers=0,
        )
    )

    evaluator_a = CNNAfterstateEvaluator.load(result["best_checkpoint"])
    evaluator_b = CNNAfterstateEvaluator.load(result["best_checkpoint"])
    board = np.array(
        [
            [1, 1, 0, 0],
            [2, 0, 0, 0],
            [3, 0, 0, 0],
            [4, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    assert evaluator_a(board) == evaluator_b(board)
    batch = np.stack([board, np.rot90(board).copy()])
    np.testing.assert_allclose(
        evaluator_a.evaluate_afterstates(batch),
        [
            evaluator_a.evaluate_afterstate(batch[0]),
            evaluator_a.evaluate_afterstate(batch[1]),
        ],
    )
    symmetry_evaluator = CNNAfterstateEvaluator.load(
        result["best_checkpoint"],
        symmetry_average=True,
    )
    np.testing.assert_allclose(
        symmetry_evaluator.evaluate_afterstates(batch),
        [
            symmetry_evaluator.evaluate_afterstate(batch[0]),
            symmetry_evaluator.evaluate_afterstate(batch[1]),
        ],
    )

    policy = SupervisedCNNPolicy(
        checkpoint=result["best_checkpoint"],
        depth=0,
    )
    batch_call_shapes = []
    original_batch_evaluate = policy.evaluator.evaluate_afterstates

    def counted_batch_evaluate(afterstates):
        batch_call_shapes.append(np.asarray(afterstates).shape)
        return original_batch_evaluate(afterstates)

    policy.evaluator.evaluate_afterstates = counted_batch_evaluate
    action = policy.predict(board)
    assert len(batch_call_shapes) == 1
    assert len(batch_call_shapes[0]) == 3
    _afterstate, _reward, is_legal = TwentyFortyEightEnv.apply_action(
        board,
        action,
    )
    assert is_legal


def test_batched_root_evaluation_matches_scalar_depth_zero_and_one():
    board = np.array(
        [
            [1, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
            [12, 13, 0, 0],
        ],
        dtype=np.uint8,
    )

    def scalar_value(afterstate):
        return float(np.sum(afterstate))

    class BatchEvaluator:
        def __init__(self):
            self.batch_sizes = []

        def evaluate_afterstates(self, afterstates):
            self.batch_sizes.append(len(afterstates))
            return np.asarray([scalar_value(afterstate) for afterstate in afterstates])

        def __call__(self, afterstate):
            raise AssertionError("batched search should not use scalar evaluator")

    for depth in (0, 1):
        evaluator = BatchEvaluator()
        batched = _evaluate_root_actions_batched(
            board,
            depth=depth,
            evaluator=evaluator,
            chance_samples=None,
        )
        scalar = evaluate_root_actions(
            board,
            depth=depth,
            evaluator=scalar_value,
            chance_samples=None,
        )

        np.testing.assert_allclose(batched.scores, scalar.scores)
        np.testing.assert_array_equal(batched.legal_mask, scalar.legal_mask)
        assert len(evaluator.batch_sizes) == 1
