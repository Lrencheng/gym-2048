from gymnasium_2048.agents.expectimax import (
    generate_expectimax_dataset,
    load_expectimax_dataset,
    save_expectimax_dataset,
)


def test_expectimax_dataset_roundtrip(tmp_path):
    path = tmp_path / "teacher_data.npz"
    dataset = generate_expectimax_dataset(
        episodes=1,
        depth=0,
        seed=5,
        max_steps=2,
        progress=False,
    )

    saved_paths = save_expectimax_dataset(dataset, path)
    loaded = load_expectimax_dataset(path)

    assert saved_paths == [path]
    assert loaded["after_boards"].shape == dataset["after_boards"].shape
    assert loaded["metadata"]["depth"] == 0
    assert loaded["metadata"]["num_samples"] == len(dataset["target_us"])


def test_expectimax_dataset_sharded_roundtrip(tmp_path):
    output_dir = tmp_path / "teacher_shards"
    dataset = generate_expectimax_dataset(
        episodes=1,
        depth=0,
        seed=7,
        max_steps=3,
        progress=False,
    )

    saved_paths = save_expectimax_dataset(dataset, output_dir, shard_size=2)
    loaded = load_expectimax_dataset(output_dir)

    assert len(saved_paths) >= 2
    assert saved_paths[0].name == "dataset_part_000.npz"
    assert len(loaded["target_us"]) == len(dataset["target_us"])
    assert loaded["metadata"]["num_samples"] == len(dataset["target_us"])
