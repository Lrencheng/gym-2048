from __future__ import annotations

import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

from gymnasium_2048.agents.config import (
    DEFAULT_ENV_ID,
    dataclass_from_mapping,
    default_config_path,
    dump_yaml_mapping,
    load_yaml_mapping,
    validate_config_agent,
)
from gymnasium_2048.agents.expectimax import (
    ExpectimaxHeuristicWeights,
    ExpectimaxPolicy,
)
from gymnasium_2048.agents.heuristic import HeuristicPolicy, HeuristicWeights
from gymnasium_2048.agents.ntuple.training import make_ntuple_policy
from gymnasium_2048.agents.supervised_cnn import SupervisedCNNPolicy


plt.style.use("ggplot")


class PredictPolicy(Protocol):
    def predict(self, state: np.ndarray) -> int:
        ...


@dataclass(frozen=True)
class EvaluationConfig:
    agent: str
    env_id: str = DEFAULT_ENV_ID
    trained_agent: str | None = None
    checkpoint: str | None = None
    episodes: int = 1000
    seed: int = 42
    title: str | None = None
    output_path: str | None = None
    plot: bool = True
    depth: int = 2
    chance_samples: int | None = None
    full_chance_empty_threshold: int = 6
    device: str = "cpu"
    weights: dict[str, float] | None = None
    reward_transform: str | None = None


def _make_heuristic_weights(data: dict[str, float] | None) -> HeuristicWeights:
    return HeuristicWeights(**data) if data is not None else HeuristicWeights()


def _make_expectimax_weights(data: dict[str, float] | None) -> ExpectimaxHeuristicWeights:
    return (
        ExpectimaxHeuristicWeights(**data)
        if data is not None
        else ExpectimaxHeuristicWeights()
    )


def make_env(env_id: str) -> gym.Env:
    env = gym.make(env_id)
    return gym.wrappers.RecordEpisodeStatistics(env)


def make_policy(config: EvaluationConfig) -> PredictPolicy | None:
    agent = config.agent
    model_path = config.checkpoint or config.trained_agent
    if agent == "heuristic":
        return HeuristicPolicy(
            weights=_make_heuristic_weights(config.weights),
            reward_transform=config.reward_transform or "raw",
        )
    if agent == "expectimax":
        return ExpectimaxPolicy(
            depth=config.depth,
            weights=_make_expectimax_weights(config.weights),
            reward_transform=config.reward_transform or "log2p1",
            seed=config.seed,
            chance_samples=config.chance_samples,
            full_chance_empty_threshold=config.full_chance_empty_threshold,
        )
    if agent == "supervised_cnn":
        if model_path is None:
            raise ValueError("supervised_cnn evaluation requires checkpoint in YAML")
        return SupervisedCNNPolicy.load(
            model_path,
            device=config.device,
            seed=config.seed,
        )
    if agent in {"ql", "tdl", "tdl-small"}:
        if model_path is None:
            raise ValueError(f"{agent} evaluation requires trained_agent in YAML")
        return make_ntuple_policy(agent=agent, trained_agent=model_path)
    return None


def _scalar(value: object) -> int:
    return int(np.asarray(value).item())


def run_episodes(
    env: gym.Env,
    policy: PredictPolicy | None,
    n_episodes: int,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int], list[int], list[int], float]:
    lengths = []
    rewards = []
    max_tiles = []
    total_score = []
    illegal_counts = []
    start_time = time.perf_counter()

    env.action_space.seed(seed)
    episode_seeds = np.random.default_rng(seed).integers(
        0,
        2**31 - 1,
        size=n_episodes,
    )

    for episode in trange(n_episodes, desc="Episode", unit="episode"):
        _observation, info = env.reset(seed=int(episode_seeds[episode]))
        terminated = truncated = False

        while not terminated and not truncated:
            action = (
                env.action_space.sample()
                if policy is None
                else policy.predict(state=info["board"])
            )
            _observation, _reward, terminated, truncated, info = env.step(action)

        lengths.append(_scalar(info["episode"]["l"]))
        rewards.append(_scalar(info["episode"]["r"]))
        max_tiles.append(int(info["max"]))
        total_score.append(int(info["total_score"]))
        illegal_counts.append(int(info["illegal_count"]))

    runtime = time.perf_counter() - start_time
    return lengths, rewards, max_tiles, total_score, illegal_counts, runtime


def plot_statistics(
    lengths: list[int],
    rewards: list[int],
    max_tiles: list[int],
    total_score: list[int],
    title: str | None = None,
) -> plt.Figure:
    fig, axs = plt.subplots(2, 2)

    axs[0, 0].hist(lengths)
    axs[0, 0].set_xlabel("Length")
    axs[0, 0].set_ylabel("Count")
    axs[0, 0].set_title("Length")

    axs[0, 1].hist(rewards)
    axs[0, 1].set_xlabel("Reward")
    axs[0, 1].set_ylabel("Count")
    axs[0, 1].set_title("Reward")

    values, counts = np.unique(max_tiles, return_counts=True)
    labels = np.power(2, values, dtype=int)
    order = np.argsort(labels)
    axs[1, 0].bar(labels[order].astype(str), counts[order])
    axs[1, 0].set_xlabel("Max number")
    axs[1, 0].set_ylabel("Count")
    axs[1, 0].set_title("Max number")

    axs[1, 1].hist(total_score)
    axs[1, 1].set_xlabel("Score")
    axs[1, 1].set_ylabel("Count")
    axs[1, 1].set_title("Score")

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def summarize_statistics(
    lengths: list[int],
    max_tiles: list[int],
    total_score: list[int],
    illegal_counts: list[int],
    runtime: float,
) -> dict[str, object]:
    actual_max_tiles = np.power(2, np.asarray(max_tiles, dtype=int), dtype=int)
    distribution = {
        int(tile): int(count)
        for tile, count in sorted(Counter(actual_max_tiles.tolist()).items())
    }
    total_steps = max(int(np.sum(lengths)), 1)
    return {
        "episodes": len(total_score),
        "mean_score": float(np.mean(total_score)),
        "best_score": int(np.max(total_score)),
        "mean_steps": float(np.mean(lengths)),
        "max_tile": int(np.max(actual_max_tiles)),
        "max_tile_distribution": distribution,
        "reach_2048_rate": float(np.mean(actual_max_tiles >= 2048)),
        "reach_4096_rate": float(np.mean(actual_max_tiles >= 4096)),
        "reach_8192_rate": float(np.mean(actual_max_tiles >= 8192)),
        "illegal_action_rate": float(np.sum(illegal_counts) / total_steps),
        "runtime_seconds": float(runtime),
    }


def print_summary(summary: dict[str, object]) -> None:
    print(
        "Evaluation summary: "
        f"episodes={summary['episodes']}, "
        f"mean_score={summary['mean_score']:.2f}, "
        f"best_score={summary['best_score']}, "
        f"mean_steps={summary['mean_steps']:.2f}, "
        f"max_tile={summary['max_tile']}, "
        f"2048_rate={summary['reach_2048_rate']:.3f}, "
        f"4096_rate={summary['reach_4096_rate']:.3f}, "
        f"8192_rate={summary['reach_8192_rate']:.3f}, "
        f"illegal_action_rate={summary['illegal_action_rate']:.5f}, "
        f"runtime_seconds={summary['runtime_seconds']:.2f}"
    )
    print(f"Max tile distribution: {summary['max_tile_distribution']}")


def load_evaluation_config(
    agent: str,
    config_path: str | Path | None = None,
) -> tuple[EvaluationConfig, dict[str, object]]:
    source = Path(config_path) if config_path is not None else default_config_path(agent, "evaluate")
    raw = validate_config_agent(load_yaml_mapping(source), agent, source)
    config = dataclass_from_mapping(EvaluationConfig, raw, source)
    return config, asdict(config)


def evaluate_config(config: EvaluationConfig) -> dict[str, object]:
    np.random.seed(config.seed)
    env = make_env(env_id=config.env_id)
    policy = make_policy(config)
    lengths, rewards, max_tiles, total_score, illegal_counts, runtime = run_episodes(
        env=env,
        policy=policy,
        n_episodes=config.episodes,
        seed=config.seed,
    )
    env.close()

    summary = summarize_statistics(
        lengths=lengths,
        max_tiles=max_tiles,
        total_score=total_score,
        illegal_counts=illegal_counts,
        runtime=runtime,
    )
    print_summary(summary)

    if config.plot or config.output_path is not None:
        fig = plot_statistics(
            lengths=lengths,
            rewards=rewards,
            max_tiles=max_tiles,
            total_score=total_score,
            title=config.title,
        )
        if config.output_path is not None:
            fig.savefig(config.output_path)
        if config.plot:
            fig.show()
    return summary


def evaluate_from_yaml(
    agent: str,
    config_path: str | Path | None = None,
    print_config: bool = False,
) -> dict[str, object] | None:
    config, printable = load_evaluation_config(
        agent=agent,
        config_path=config_path,
    )
    if print_config:
        print(dump_yaml_mapping(printable))
        return None
    return evaluate_config(config)


__all__ = [
    "EvaluationConfig",
    "PredictPolicy",
    "evaluate_config",
    "evaluate_from_yaml",
    "load_evaluation_config",
    "make_env",
    "make_policy",
    "plot_statistics",
    "print_summary",
    "run_episodes",
    "summarize_statistics",
]
