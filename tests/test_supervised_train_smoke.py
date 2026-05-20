from gymnasium_2048.agents.expectimax import (
    generate_expectimax_dataset,
    save_expectimax_dataset,
)
from gymnasium_2048.agents.supervised_cnn import (
    SupervisedTrainingConfig,
    train_supervised_cnn,
)


def test_supervised_train_smoke(tmp_path):
    data_path = tmp_path / "teacher_data.npz"
    out_dir = tmp_path / "cnn"
    dataset = generate_expectimax_dataset(
        episodes=1,
        depth=1,
        seed=9,
        max_steps=4,
        progress=False,
    )
    save_expectimax_dataset(dataset, data_path)

    result = train_supervised_cnn(
        SupervisedTrainingConfig(
            data_path=str(data_path),
            out_dir=str(out_dir),
            epochs=1,
            batch_size=2,
            seed=9,
            use_high_score_weighting=True,
        )
    )

    assert (out_dir / "best.pt").exists()
    assert (out_dir / "last.pt").exists()
    assert result["best_validation_loss"] >= 0.0
