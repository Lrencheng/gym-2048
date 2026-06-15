import numpy as np

from gymnasium_2048.agents.expectimax import (
    augment_afterstate_samples,
    generate_expectimax_dataset,
)


def test_expectimax_data_generation_emits_afterstate_value_samples():
    dataset = generate_expectimax_dataset(
        episodes=1,
        depth=0,
        seed=3,
        max_steps=2,
        progress=False,
        workers=1,
        debug_fields=True,
    )

    sample_count = len(dataset["target_us"])
    assert sample_count >= 2
    assert dataset["after_boards"].shape == (sample_count, 4, 4)
    assert dataset["after_boards"].dtype == np.uint8
    assert dataset["target_us"].shape == (sample_count,)
    assert dataset["target_us"].dtype == np.float32
    assert dataset["immediate_rewards"].dtype == np.float32
    assert dataset["actions"].dtype == np.int64
    assert dataset["depths"].dtype == np.int64
    assert dataset["root_ids"].dtype == np.int64
    assert dataset["episodes"].dtype == np.int64
    assert dataset["root_boards"].shape == (sample_count, 4, 4)
    np.testing.assert_allclose(
        dataset["target_qs"],
        dataset["immediate_rewards"] + dataset["target_us"],
    )
    assert dataset["metadata"]["target"] == "afterstate_continuation_value"
    assert dataset["metadata"]["board_format"] == "exponent"


def test_expectimax_data_generation_workers_smoke():
    dataset = generate_expectimax_dataset(
        episodes=2,
        depth=0,
        seed=13,
        max_steps=1,
        progress=False,
        workers=2,
    )

    assert set(dataset["episodes"].tolist()) == {0, 1}
    assert dataset["metadata"]["workers"] == 2
    assert dataset["metadata"]["num_roots"] == 2


def test_afterstate_augmentation_preserves_target_and_transforms_actions():
    dataset = generate_expectimax_dataset(
        episodes=1,
        depth=0,
        seed=17,
        max_steps=1,
        progress=False,
        workers=1,
    )

    augmented = augment_afterstate_samples(dataset)

    assert len(augmented["target_us"]) == 8 * len(dataset["target_us"])
    np.testing.assert_array_equal(
        augmented["target_us"].reshape(-1, 8),
        np.repeat(dataset["target_us"][:, None], 8, axis=1),
    )
    np.testing.assert_array_equal(
        augmented["root_ids"].reshape(-1, 8),
        np.repeat(dataset["root_ids"][:, None], 8, axis=1),
    )
    assert np.all((0 <= augmented["actions"]) & (augmented["actions"] < 4))
    assert augmented["metadata"]["symmetry_augmentation"] == "all_8"
