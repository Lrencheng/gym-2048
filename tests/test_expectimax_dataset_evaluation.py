from gymnasium_2048.agents.expectimax import (
    generate_expectimax_dataset,
    save_expectimax_dataset,
)
from gymnasium_2048.agents.expectimax.evaluate_dataset import (
    summarize_expectimax_dataset,
)


def test_expectimax_dataset_evaluation_summary(tmp_path):
    path = tmp_path / "teacher_data.npz"
    dataset = generate_expectimax_dataset(
        episodes=2,
        depth=0,
        seed=21,
        max_steps=2,
        progress=False,
    )
    save_expectimax_dataset(dataset, path)

    summary = summarize_expectimax_dataset(path)

    assert summary["episodes"] == 2
    assert summary["roots"] == 4
    assert summary["samples"] == len(dataset["target_us"])
    assert summary["target_u"]["maximum"] >= summary["target_u"]["minimum"]
    assert summary["immediate_reward"]["minimum"] >= 0.0
