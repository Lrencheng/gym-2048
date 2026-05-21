from pathlib import Path

import gymnasium as gym
import pytest

from gymnasium_2048.agents.config import ConfigError, default_config_path
from gymnasium_2048.agents.evaluation import EvaluationConfig, make_policy, run_episodes
from gymnasium_2048.agents.supervised_cnn.config import (
    load_supervised_training_config,
    resolve_supervised_output_dir,
)
from scripts.evaluate import parse_args as parse_evaluate_args
from scripts.train import parse_args as parse_train_args


def test_evaluate_expectimax_integration_smoke():
    env = gym.make("gymnasium_2048:gymnasium_2048/TwentyFortyEight-v0")
    env = gym.wrappers.RecordEpisodeStatistics(env)
    policy = make_policy(
        EvaluationConfig(
            agent="expectimax",
            depth=1,
            seed=13,
            episodes=1,
        )
    )

    lengths, _rewards, _max_tiles, scores, illegal_counts, _runtime = run_episodes(
        env=env,
        policy=policy,
        n_episodes=1,
        seed=13,
    )
    env.close()

    assert len(lengths) == 1
    assert len(scores) == 1
    assert illegal_counts[0] == 0


def test_train_parser_is_generic(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--agent",
            "supervised_cnn",
            "--config",
            "config.yaml",
            "--print-config",
        ],
    )

    args = parse_train_args()

    assert args.agent == "supervised_cnn"
    assert args.config == "config.yaml"
    assert args.print_config is True
    assert not hasattr(args, "epochs")
    assert not hasattr(args, "batch_size")


def test_evaluate_parser_is_generic(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate.py",
            "--agent",
            "expectimax",
            "--config",
            "evaluate.yaml",
            "--print-config",
        ],
    )

    args = parse_evaluate_args()

    assert args.agent == "expectimax"
    assert args.config == "evaluate.yaml"
    assert args.print_config is True
    assert not hasattr(args, "depth")
    assert not hasattr(args, "episodes")


def test_supervised_default_config_loads_without_creating_root(tmp_path, monkeypatch):
    import gymnasium_2048.agents.supervised_cnn.config as supervised_config

    root = tmp_path / "models" / "supervise"
    monkeypatch.setattr(supervised_config, "SUPERVISED_RUN_ROOT", root)

    config, printable = load_supervised_training_config(
        default_config_path("supervised_cnn", "train")
    )

    assert config.data_path == "data/expectimax_d2_sampled_1000eps.npz"
    assert config.epochs == 40
    assert config.batch_size == 2048
    assert config.learning_rate == 0.001
    assert config.weight_decay == 0.0001
    assert config.temperature == 1.0
    assert config.device == "cuda"
    assert config.num_workers == 4
    assert config.encode_on_device is True
    assert config.validation_max_samples == 100000
    assert config.amp is True
    resolved = Path(printable["resolved_out_dir"])
    assert resolved.parts[-3:-1] == ("models", "supervise")
    assert resolved.name.startswith("train")


def test_config_agent_mismatch_raises(tmp_path):
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("agent: expectimax\n", encoding="utf-8")

    with pytest.raises(ConfigError):
        load_supervised_training_config(config_path)


def test_supervised_output_dir_is_kept_under_models():
    assert resolve_supervised_output_dir(
        out_dir="checkpoints/supervised_cnn",
        create_root=False,
    ) == Path("models") / "supervise" / "supervised_cnn"
    assert resolve_supervised_output_dir(
        out_dir="custom_run",
        create_root=False,
    ) == Path("models") / "supervise" / "custom_run"


def test_supervised_output_dir_auto_increments(tmp_path):
    root = tmp_path / "models" / "supervise"
    (root / "train1").mkdir(parents=True)

    assert resolve_supervised_output_dir(out_root=root, create_root=False) == root / "train2"
    assert (
        resolve_supervised_output_dir(
            out_root=root,
            out_dir="models",
            create_root=False,
        )
        == root / "train2"
    )
