from gymnasium_2048.agents.expectimax import (
    generate_expectimax_dataset,
    save_expectimax_dataset,
)
from gymnasium_2048.agents.expectimax.evaluate_dataset import summarize_expectimax_dataset


def test_expectimax_dataset_evaluation_summary(tmp_path):
    path = tmp_path / "teacher_data.npz"
    dataset = generate_expectimax_dataset(
        episodes=2,
        depth=1,
        seed=21,
        max_steps=2,
        progress=False,
    )
    save_expectimax_dataset(dataset, path)

    summary = summarize_expectimax_dataset(path)

    assert summary["episodes"] == 2
    assert summary["samples"] == 4
    assert summary["labels"]["illegal_action_labels"] == 0
    assert summary["labels"]["empty_legal_masks"] == 0
    assert "2048" in summary["reach_rates"]
