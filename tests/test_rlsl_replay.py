import numpy as np
import torch

from gymnasium_2048.agents.RLSL import EpisodeSample, RLSLTrainingConfig
from gymnasium_2048.agents.RLSL.replay import (
    REPLAY_SOURCE_ONLINE,
    REPLAY_SOURCE_TEACHER,
    RLSLReplayBuffer,
    ReplayAfterstateDataset,
    cap_current_samples,
    choose_admitted_samples,
)


def _sample(value: int, target: float | None = None) -> EpisodeSample:
    return EpisodeSample(
        afterstate=np.full((4, 4), value, dtype=np.uint8),
        target_value=float(value if target is None else target),
    )


def test_replay_buffer_initializes_from_teacher_dataset_with_capacity():
    teacher_boards = np.arange(6 * 16, dtype=np.uint8).reshape(6, 4, 4)
    teacher_targets = np.arange(6, dtype=np.float32)

    buffer = RLSLReplayBuffer.from_arrays(
        teacher_boards,
        teacher_targets,
        capacity=4,
        seed=123,
    )

    assert len(buffer) == 4
    assert buffer.source_counts() == {
        REPLAY_SOURCE_TEACHER: 4,
        REPLAY_SOURCE_ONLINE: 0,
    }
    boards, targets, sources = buffer.arrays()
    assert boards.shape == (4, 4, 4)
    assert targets.shape == (4,)
    assert np.all(sources == REPLAY_SOURCE_TEACHER)


def test_replay_buffer_fifo_append_replaces_oldest_teacher_samples():
    teacher_boards = np.stack(
        [np.full((4, 4), value, dtype=np.uint8) for value in [1, 2, 3]]
    )
    teacher_targets = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    buffer = RLSLReplayBuffer.from_arrays(
        teacher_boards,
        teacher_targets,
        capacity=3,
        seed=0,
        shuffle=False,
    )

    buffer.append_online([_sample(10), _sample(11)])

    boards, targets, sources = buffer.arrays()
    assert targets.tolist() == [3.0, 10.0, 11.0]
    assert boards[:, 0, 0].tolist() == [3, 10, 11]
    assert sources.tolist() == [
        REPLAY_SOURCE_TEACHER,
        REPLAY_SOURCE_ONLINE,
        REPLAY_SOURCE_ONLINE,
    ]
    assert buffer.source_counts() == {
        REPLAY_SOURCE_TEACHER: 1,
        REPLAY_SOURCE_ONLINE: 2,
    }


def test_current_samples_are_capped_reproducibly():
    samples = [_sample(value) for value in range(10)]

    first = cap_current_samples(samples, max_samples=4, rng=np.random.default_rng(7))
    second = cap_current_samples(samples, max_samples=4, rng=np.random.default_rng(7))

    assert [sample.target_value for sample in first] == [
        sample.target_value for sample in second
    ]
    assert len(first) == 4
    assert set(sample.target_value for sample in first).issubset(set(range(10)))


def test_admission_fraction_controls_fifo_write_count():
    samples = [_sample(value) for value in range(10)]

    admitted = choose_admitted_samples(
        samples,
        fraction=0.25,
        rng=np.random.default_rng(11),
    )

    assert len(admitted) == 2
    assert set(sample.target_value for sample in admitted).issubset(set(range(10)))


def test_replay_afterstate_dataset_includes_buffer_and_current_samples():
    teacher_boards = np.stack(
        [np.full((4, 4), value, dtype=np.uint8) for value in [1, 2]]
    )
    teacher_targets = np.asarray([1.0, 2.0], dtype=np.float32)
    buffer = RLSLReplayBuffer.from_arrays(
        teacher_boards,
        teacher_targets,
        capacity=4,
        seed=0,
        shuffle=False,
    )
    buffer.append_online([_sample(3)])
    current = [_sample(4), _sample(5)]
    config = RLSLTrainingConfig(
        out_dir="unused",
        input_channels=16,
        symmetry_augmentation=False,
        progress=False,
    )

    dataset = ReplayAfterstateDataset(
        buffer=buffer,
        current_samples=current,
        config=config,
        seed=123,
    )

    assert len(dataset) == 5
    encoded, target = dataset[4]
    assert isinstance(encoded, torch.Tensor)
    assert encoded.shape == (16, 4, 4)
    assert target.item() == 5.0
