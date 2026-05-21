from gymnasium_2048.agents.expectimax import (
    generate_expectimax_dataset,
    save_expectimax_dataset,
)
from gymnasium_2048.agents.supervised_cnn import (
    SupervisedTrainingConfig,
    resolve_device,
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
            device="cpu",
            use_high_score_weighting=True,
            num_workers=0,
            encode_on_device=True,
            amp=False,
            profile=True,
            profile_batches=1,
            validation_max_samples=2,
        )
    )

    assert (out_dir / "checkpoints" / "best.pt").exists()
    assert (out_dir / "checkpoints" / "last.pt").exists()
    assert (out_dir / "tables" / "history.csv").exists()
    assert (out_dir / "plots" / "training_curves.png").exists()
    assert (out_dir / "tables" / "dataset_split_summary.csv").exists()
    assert (out_dir / "plots" / "dataset_analysis.png").exists()
    assert (out_dir / "data" / "teacher_data.npz").exists()
    assert (out_dir / "data" / "train_indices.npy").exists()
    assert (out_dir / "data" / "validation_indices.npy").exists()
    assert result["best_validation_loss"] >= 0.0
    assert result["output_dir"] == str(out_dir)
    assert result["history"][0]["train_samples_per_second"] > 0.0
    assert result["history"][0]["profiled_batches"] == 1.0


def test_supervised_train_smoke_with_workers(tmp_path):
    data_path = tmp_path / "teacher_data_workers.npz"
    out_dir = tmp_path / "cnn_workers"
    dataset = generate_expectimax_dataset(
        episodes=1,
        depth=1,
        seed=19,
        max_steps=3,
        progress=False,
    )
    save_expectimax_dataset(dataset, data_path)

    result = train_supervised_cnn(
        SupervisedTrainingConfig(
            data_path=str(data_path),
            out_dir=str(out_dir),
            epochs=1,
            batch_size=2,
            seed=19,
            device="cpu",
            num_workers=2,
            persistent_workers=True,
            prefetch_factor=2,
            encode_on_device=True,
            amp=False,
            copy_training_data=False,
            validation_max_samples=1,
        )
    )

    assert (out_dir / "checkpoints" / "best.pt").exists()
    assert result["history"][0]["validation_samples"] == 1


def test_resolve_device_auto_returns_valid_torch_device():
    device = resolve_device("auto")

    assert device.type in {"cpu", "cuda"}
