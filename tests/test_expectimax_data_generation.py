from gymnasium_2048.agents.expectimax import generate_expectimax_dataset


def test_expectimax_data_generation_smoke():
    dataset = generate_expectimax_dataset(
        episodes=1,
        depth=1,
        seed=3,
        max_steps=3,
        progress=False,
        chance_samples=6,
        workers=1,
    )

    assert dataset["boards"].shape[0] == 3
    assert dataset["legal_masks"].shape == (3, 4)
    assert dataset["action_scores"].shape == (3, 4)
    assert dataset["action_probs"].shape == (3, 4)
    assert dataset["metadata"]["board_format"] == "exponent"
    assert dataset["metadata"]["chance_samples"] == 6
    assert dataset["metadata"]["full_chance_empty_threshold"] == 6
    assert dataset["metadata"]["workers"] == 1


def test_expectimax_data_generation_workers_smoke():
    dataset = generate_expectimax_dataset(
        episodes=2,
        depth=1,
        seed=13,
        max_steps=2,
        progress=False,
        workers=2,
    )

    assert dataset["boards"].shape[0] == 4
    assert dataset["episodes"].tolist() == [0, 0, 1, 1]
    assert dataset["metadata"]["workers"] == 2
