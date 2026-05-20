from gymnasium_2048.agents.expectimax import (
    generate_expectimax_dataset,
    load_expectimax_dataset,
    save_expectimax_dataset,
)


def test_expectimax_dataset_roundtrip(tmp_path):
    path = tmp_path / "teacher_data.npz"
    dataset = generate_expectimax_dataset(
        episodes=1,
        depth=1,
        seed=5,
        max_steps=2,
        progress=False,
    )

    save_expectimax_dataset(dataset, path)
    loaded = load_expectimax_dataset(path)

    assert loaded["boards"].shape[0] == 2
    assert loaded["metadata"]["depth"] == 1
    assert loaded["metadata"]["num_samples"] == 2
