import numpy as np
import pytest
import torch

from gymnasium_2048.agents.config import default_config_path
from gymnasium_2048.agents.registry import TRAIN_AGENTS
import gymnasium_2048.agents.RLSL.train as rlsl_train
from gymnasium_2048.agents.RLSL import (
    EpisodeSample,
    EpisodeRollout,
    RLSLTrainingConfig,
    SearchImprovedActionSample,
    load_rlsl_training_config,
    train_on_episode_samples,
)
from gymnasium_2048.agents.supervised_cnn import CNNConfig, SupervisedCNN
from gymnasium_2048.agents.supervised_cnn.model import config_to_dict


def test_rlsl_default_config_loads_and_agent_is_registered():
    config, printable = load_rlsl_training_config(default_config_path("RLSL", "train"))

    assert "RLSL" in TRAIN_AGENTS
    assert config.search_depth == 1
    assert config.train_epochs_per_episode == 1
    assert config.batch_size == 8900
    assert config.learning_rate <= 1.0e-4
    assert config.replay_enabled
    assert config.replay_capacity == 280_000
    assert config.replay_current_train_max_samples == 70_000
    assert config.replay_current_admission_fraction == pytest.approx(0.10)
    assert config.loss_type in {"mse", "huber"}
    assert printable["resolved_out_dir"].startswith("models")


def test_episode_supervised_update_reports_explicit_l2_component():
    model = SupervisedCNN(CNNConfig(input_channels=16, conv_channels=4, hidden_size=8))
    for parameter in model.parameters():
        torch.nn.init.constant_(parameter, 0.1)

    samples = [
        EpisodeSample(afterstate=np.zeros((4, 4), dtype=np.uint8), target_value=0.0),
        EpisodeSample(afterstate=np.ones((4, 4), dtype=np.uint8), target_value=1.0),
    ]
    config = RLSLTrainingConfig(
        out_dir="unused",
        train_epochs_per_episode=1,
        batch_size=2,
        learning_rate=0.0,
        l2_coeff=0.25,
        loss_type="mse",
        value_scale=1.0,
        symmetry_augmentation=False,
        replay_enabled=False,
        progress=False,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

    stats = train_on_episode_samples(
        model=model,
        samples=samples,
        config=config,
        device=torch.device("cpu"),
        optimizer=optimizer,
    )

    assert stats["l2_loss"] > 0.0
    assert stats["total_loss"] == pytest.approx(
        stats["regression_loss"] + stats["l2_loss"]
    )
    assert stats["samples"] == 2


def test_episode_supervised_update_uses_fixed_target_normalization():
    model = SupervisedCNN(CNNConfig(input_channels=16, conv_channels=4, hidden_size=8))
    for parameter in model.parameters():
        torch.nn.init.constant_(parameter, 0.0)

    samples = [
        EpisodeSample(afterstate=np.zeros((4, 4), dtype=np.uint8), target_value=10.0),
        EpisodeSample(afterstate=np.ones((4, 4), dtype=np.uint8), target_value=10.0),
    ]
    config = RLSLTrainingConfig(
        out_dir="unused",
        train_epochs_per_episode=1,
        batch_size=2,
        learning_rate=0.0,
        weight_decay=0.0,
        l2_coeff=0.0,
        loss_type="mse",
        value_scale=1.0,
        target_mean=10.0,
        target_std=2.0,
        symmetry_augmentation=False,
        replay_enabled=False,
        progress=False,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

    stats = train_on_episode_samples(
        model=model,
        samples=samples,
        config=config,
        device=torch.device("cpu"),
        optimizer=optimizer,
    )

    assert stats["regression_loss"] == pytest.approx(0.0)


def test_model_initialization_inherits_checkpoint_target_normalization(tmp_path):
    model = SupervisedCNN(CNNConfig(input_channels=16, conv_channels=4, hidden_size=8))
    checkpoint = tmp_path / "initial.pt"
    torch.save(
        {
            "model_config": config_to_dict(model.config),
            "model_state_dict": model.state_dict(),
            "target_mean": 7.5,
            "target_std": 3.25,
        },
        checkpoint,
    )
    config = RLSLTrainingConfig(
        out_dir="unused",
        initial_checkpoint=str(checkpoint),
        conv_channels=4,
        hidden_size=8,
        replay_enabled=False,
        progress=False,
    )

    _model, normalization = rlsl_train._make_model(config, torch.device("cpu"))

    assert normalization.mean == pytest.approx(7.5)
    assert normalization.std == pytest.approx(3.25)


def test_episode_collection_stores_all_legal_root_action_samples(monkeypatch):
    class FakeEnv:
        def reset(self, seed):
            return None, {
                "board": np.zeros((4, 4), dtype=np.uint8),
                "total_score": 0,
            }

        def step(self, action):
            return None, 0, True, False, {
                "board": np.ones((4, 4), dtype=np.uint8),
                "total_score": 4,
            }

        def close(self):
            pass

    action_samples = (
        SearchImprovedActionSample(
            action=0,
            afterstate=np.zeros((4, 4), dtype=np.uint8),
            target_value=1.0,
            root_score=1.0,
            immediate_reward=0.0,
        ),
        SearchImprovedActionSample(
            action=1,
            afterstate=np.ones((4, 4), dtype=np.uint8),
            target_value=2.0,
            root_score=6.0,
            immediate_reward=4.0,
        ),
    )

    def fake_choose_search_improved_action(*_args, **_kwargs):
        return type(
            "Decision",
            (),
            {
                "legal_mask": np.asarray([True, True, False, False]),
                "action": 1,
                "action_samples": action_samples,
            },
        )()

    monkeypatch.setattr(rlsl_train.gym, "make", lambda _env_id: FakeEnv())
    monkeypatch.setattr(
        rlsl_train,
        "choose_search_improved_action",
        fake_choose_search_improved_action,
    )
    config = RLSLTrainingConfig(out_dir="unused", replay_enabled=False, progress=False)

    rollout = rlsl_train.collect_search_improved_episode(
        config,
        evaluator=object(),
        episode_seed=123,
        policy_rng=np.random.default_rng(123),
    )

    assert rollout.length == 1
    assert rollout.score == 4
    assert len(rollout.samples) == len(action_samples)
    assert [sample.target_value for sample in rollout.samples] == [1.0, 2.0]


def test_best_checkpoint_updates_only_from_evaluation_metric(monkeypatch, tmp_path):
    rollouts = [
        EpisodeRollout(
            samples=[],
            score=10_000,
            max_tile=2048,
            length=1000,
        ),
        EpisodeRollout(
            samples=[],
            score=100,
            max_tile=128,
            length=100,
        ),
    ]
    best_rounds = []

    def fake_collect(*_args, **_kwargs):
        return rollouts.pop(0)

    def fake_train_on_episode_samples(**_kwargs):
        return {
            "samples": 0,
            "epochs": 1,
            "regression_loss": 0.0,
            "l2_loss": 0.0,
            "total_loss": 0.0,
            "train_loss": 0.0,
        }

    def fake_save_checkpoint(path, *, round_index, **_kwargs):
        if path.name == "best.pt":
            best_rounds.append(round_index)

    monkeypatch.setattr(rlsl_train, "collect_search_improved_episode", fake_collect)
    monkeypatch.setattr(
        rlsl_train,
        "train_on_episode_samples",
        fake_train_on_episode_samples,
    )
    monkeypatch.setattr(
        rlsl_train,
        "_evaluate_round",
        lambda **_kwargs: {
            "mean_score": 60.0,
            "episodes": 2,
            "best_score": 100,
            "mean_steps": 10.0,
            "max_tile": 128,
            "max_tile_distribution": {128: 2},
            "reach_2048_rate": 0.0,
            "reach_4096_rate": 0.0,
            "reach_8192_rate": 0.0,
            "illegal_action_rate": 0.0,
            "runtime_seconds": 0.0,
        },
    )
    monkeypatch.setattr(rlsl_train, "_save_checkpoint", fake_save_checkpoint)

    config = RLSLTrainingConfig(
        out_dir=str(tmp_path),
        rounds=2,
        eval_interval=2,
        eval_episodes=2,
        checkpoint_interval=1,
        conv_channels=4,
        hidden_size=8,
        replay_enabled=False,
        progress=False,
    )

    rlsl_train.train_rlsl(config)

    assert best_rounds == [2]


def test_replay_training_uses_buffer_plus_current_before_admission(
    monkeypatch,
    tmp_path,
):
    trained_sizes = []
    replay_sizes_seen_by_training = []
    history_writes = []

    class FakeReplayBuffer:
        def __init__(self):
            self.online_appended = 0

        def __len__(self):
            return 5 + self.online_appended

        def source_counts(self):
            return {0: 5, 1: self.online_appended}

        def append_online(self, samples):
            self.online_appended += len(samples)

    class FakeReplayDataset:
        def __init__(self, *, buffer, current_samples, config, seed):
            replay_sizes_seen_by_training.append(len(buffer))
            self.size = len(buffer) + len(current_samples)

        def __len__(self):
            return self.size

    rollout = EpisodeRollout(
        samples=[
            EpisodeSample(
                afterstate=np.full((4, 4), value, dtype=np.uint8),
                target_value=float(value),
            )
            for value in range(4)
        ],
        score=128,
        max_tile=64,
        length=1,
    )

    def fake_train_on_afterstate_dataset(**kwargs):
        dataset = kwargs["dataset"]
        trained_sizes.append(len(dataset))
        return {
            "samples": len(dataset),
            "epochs": 1,
            "regression_loss": 0.0,
            "l2_loss": 0.0,
            "total_loss": 0.0,
            "train_loss": 0.0,
        }

    def fake_write_json(path, data):
        if path.name == "history.json":
            history_writes.append(data)

    monkeypatch.setattr(rlsl_train, "_make_replay_buffer", lambda config: FakeReplayBuffer())
    monkeypatch.setattr(rlsl_train, "ReplayAfterstateDataset", FakeReplayDataset)
    monkeypatch.setattr(
        rlsl_train,
        "collect_search_improved_episode",
        lambda *_args, **_kwargs: rollout,
    )
    monkeypatch.setattr(
        rlsl_train,
        "train_on_afterstate_dataset",
        fake_train_on_afterstate_dataset,
    )
    monkeypatch.setattr(
        rlsl_train,
        "_evaluate_round",
        lambda **_kwargs: {
            "mean_score": 1.0,
            "episodes": 1,
            "best_score": 1,
            "mean_steps": 1.0,
            "max_tile": 2,
            "max_tile_distribution": {2: 1},
            "reach_2048_rate": 0.0,
            "reach_4096_rate": 0.0,
            "reach_8192_rate": 0.0,
            "illegal_action_rate": 0.0,
            "runtime_seconds": 0.0,
        },
    )
    monkeypatch.setattr(rlsl_train, "_write_json", fake_write_json)

    config = RLSLTrainingConfig(
        out_dir=str(tmp_path),
        rounds=1,
        replay_enabled=True,
        replay_capacity=5,
        replay_current_train_max_samples=3,
        replay_current_admission_fraction=0.5,
        checkpoint_interval=1,
        eval_interval=0,
        conv_channels=4,
        hidden_size=8,
        progress=False,
    )

    rlsl_train.train_rlsl(config)

    assert replay_sizes_seen_by_training == [5]
    assert trained_sizes == [8]
    assert history_writes[-1][0]["replay_size"] == 7
    assert history_writes[-1][0]["current_raw_samples"] == 4
    assert history_writes[-1][0]["current_train_samples"] == 3
    assert history_writes[-1][0]["current_admitted_samples"] == 2
    assert history_writes[-1][0]["train_dataset_size"] == 8
