from pathlib import Path

import gymnasium as gym
import pytest

from gymnasium_2048.agents.config import ConfigError, default_config_path
from gymnasium_2048.agents.evaluation import EvaluationConfig, make_policy, run_episodes
from gymnasium_2048.agents.registry import EVALUATE_AGENTS, TRAIN_AGENTS
from gymnasium_2048.agents.supervised_cnn.config import (
    load_supervised_training_config,
    resolve_supervised_output_dir,
)
from gymnasium_2048.agents.supervised_ntuple.config import (
    load_supervised_ntuple_config,
)
from scripts.evaluate import parse_args as parse_evaluate_args
from scripts.train import parse_args as parse_train_args


def test_evaluate_expectimax_integration_smoke():
    env = gym.make("gymnasium_2048:gymnasium_2048/TwentyFortyEight-v0")
    env = gym.wrappers.RecordEpisodeStatistics(env)
    policy = make_policy(
        EvaluationConfig(
            agent="expectimax",
            depth=0,
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


def test_supervised_agents_are_registered():
    assert {"supervised_cnn", "supervised_ntuple"} <= TRAIN_AGENTS
    assert {"supervised_cnn", "supervised_ntuple"} <= EVALUATE_AGENTS


def test_train_parser_is_generic(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--agent",
            "supervised_ntuple",
            "--config",
            "config.yaml",
            "--print-config",
        ],
    )

    args = parse_train_args()

    assert args.agent == "supervised_ntuple"
    assert args.config == "config.yaml"
    assert args.print_config is True
    assert not hasattr(args, "epochs")


def test_evaluate_parser_is_generic(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate.py",
            "--agent",
            "supervised_cnn",
            "--config",
            "evaluate.yaml",
            "--print-config",
        ],
    )

    args = parse_evaluate_args()

    assert args.agent == "supervised_cnn"
    assert args.config == "evaluate.yaml"
    assert args.print_config is True
    assert not hasattr(args, "depth")


def test_supervised_cnn_default_config_loads():
    config, printable = load_supervised_training_config(
        default_config_path("supervised_cnn", "train")
    )

    assert config.data_path == "data/expectimax_afterstates"
    assert config.epochs == 40
    assert config.loss == "huber"
    assert config.target_normalization is True
    assert config.symmetry_augmentation is True
    assert printable["resolved_out_dir"].startswith("models")


def test_supervised_ntuple_default_config_loads():
    config, printable = load_supervised_ntuple_config(
        default_config_path("supervised_ntuple", "train")
    )

    assert config.pattern_set == "rows_cols"
    assert config.target_normalization is True
    assert printable["agent"] == "supervised_ntuple"


def test_config_agent_mismatch_raises(tmp_path):
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("agent: expectimax\n", encoding="utf-8")

    with pytest.raises(ConfigError):
        load_supervised_training_config(config_path)


def test_supervised_output_dir_is_kept_under_models():
    assert resolve_supervised_output_dir(
        out_root="models/supervised_cnn",
        out_dir="custom_run",
        create_root=False,
    ) == Path("models") / "supervised_cnn" / "custom_run"


def test_supervised_output_dir_auto_increments(tmp_path):
    root = tmp_path / "models" / "supervised_cnn"
    (root / "train1").mkdir(parents=True)

    assert resolve_supervised_output_dir(
        out_root=root,
        create_root=False,
    ) == root / "train2"
