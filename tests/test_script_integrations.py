from pathlib import Path
from concurrent.futures import Future

import gymnasium as gym
import pytest

from gymnasium_2048.agents.config import ConfigError, default_config_path
from gymnasium_2048.agents import evaluation
from gymnasium_2048.agents.evaluation import (
    EvaluationConfig,
    make_policy,
    run_episodes,
    run_episodes_parallel,
)
from gymnasium_2048.agents.registry import EVALUATE_AGENTS, TRAIN_AGENTS
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


def test_parallel_evaluation_progress_reports_episodes(monkeypatch):
    progress_bars = []

    class FakeTqdm:
        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable
            self.kwargs = kwargs
            self.n = 0
            progress_bars.append(self)

        def __iter__(self):
            for item in self.iterable:
                self.update(1)
                yield item

        def update(self, count=1):
            self.n += count

        def close(self):
            pass

    class FakeExecutor:
        def __init__(self, max_workers):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def map(self, fn, tasks):
            return [fn(task) for task in tasks]

        def submit(self, fn, task):
            future = Future()
            future.set_result(fn(task))
            return future

    def fake_worker(task):
        for _seed in task.episode_seeds:
            if task.progress_queue is not None:
                task.progress_queue.put(1)
        count = len(task.episode_seeds)
        return {
            "lengths": [1] * count,
            "rewards": [0] * count,
            "max_tiles": [1] * count,
            "total_score": [0] * count,
            "illegal_counts": [0] * count,
        }

    monkeypatch.setattr(evaluation, "tqdm", FakeTqdm)
    monkeypatch.setattr(evaluation, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(evaluation, "_run_eval_worker", fake_worker)

    run_episodes_parallel(
        env_id="gymnasium_2048:gymnasium_2048/TwentyFortyEight-v0",
        config=EvaluationConfig(agent="random"),
        n_episodes=5,
        seed=13,
        workers=2,
        progress=True,
    )

    assert progress_bars[0].kwargs["total"] == 5
    assert progress_bars[0].kwargs["unit"] == "episode"
    assert progress_bars[0].n == 5


def test_supervised_agents_are_registered():
    removed_agent = "supervised" + "_ntuple"
    assert "supervised_cnn" in TRAIN_AGENTS
    assert "supervised_cnn" in EVALUATE_AGENTS
    assert removed_agent not in TRAIN_AGENTS
    assert removed_agent not in EVALUATE_AGENTS


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

    assert config.data_path == "data/expectimax_afterstates_400.npz"
    assert config.epochs == 16
    assert config.loss == "huber"
    assert config.target_normalization is True
    assert config.symmetry_augmentation is True
    assert printable["resolved_out_dir"].startswith("models")


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
