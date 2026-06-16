from __future__ import annotations

import queue
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from multiprocessing import Manager
from pathlib import Path
from typing import Any, Protocol

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm, trange

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
    symmetry_average: bool = False
    workers: int = 1


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
            depth=config.depth,
            device=config.device,
            seed=config.seed,
            chance_samples=config.chance_samples,
            full_chance_empty_threshold=config.full_chance_empty_threshold,
            symmetry_average=config.symmetry_average,
        )
    if agent in {"ql", "tdl", "tdl-small"}:
        if model_path is None:
            raise ValueError(f"{agent} evaluation requires trained_agent in YAML")
        return make_ntuple_policy(agent=agent, trained_agent=model_path)
    return None


def _scalar(value: object) -> int:
    return int(np.asarray(value).item())


@dataclass(frozen=True)
class _EvalWorkerTask:
    """Picklable task describing a worker's slice of evaluation episodes."""
    env_id: str
    episode_seeds: list[int]
    config_dict: dict[str, Any]
    progress_queue: Any | None = None


def _run_eval_worker(task: _EvalWorkerTask) -> dict[str, list[int]]:
    """Module-level worker for ProcessPoolExecutor; must be picklable."""
    env = gym.make(task.env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    config = EvaluationConfig(**task.config_dict)
    policy = make_policy(config)

    lengths = []
    rewards = []
    max_tiles = []
    total_score = []
    illegal_counts = []

    env.action_space.seed(task.episode_seeds[0])

    for seed in task.episode_seeds:
        _observation, info = env.reset(seed=seed)
        terminated = truncated = False
        while not terminated and not truncated:
            action = (
                env.action_space.sample()
                if policy is None
                else policy.predict(state=info["board"])
            )
            _observation, _reward, terminated, truncated, info = env.step(action)
        lengths.append(int(info["episode"]["l"]))
        rewards.append(int(info["episode"]["r"]))
        max_tiles.append(int(info["max"]))
        total_score.append(int(info["total_score"]))
        illegal_counts.append(int(info["illegal_count"]))
        if task.progress_queue is not None:
            task.progress_queue.put(1)

    env.close()
    return {
        "lengths": lengths,
        "rewards": rewards,
        "max_tiles": max_tiles,
        "total_score": total_score,
        "illegal_counts": illegal_counts,
    }


def run_episodes_parallel(
    env_id: str,
    config: EvaluationConfig,
    n_episodes: int,
    seed: int = 42,
    workers: int = 1,
    progress: bool = True,
) -> tuple[list[int], list[int], list[int], list[int], list[int], float]:
    """Evaluate episodes in parallel across multiple worker processes."""
    start_time = time.perf_counter()

    rng = np.random.default_rng(seed)
    all_episode_seeds = rng.integers(0, 2**31 - 1, size=n_episodes).tolist()

    chunk_size = (n_episodes + workers - 1) // workers
    seed_chunks = [
        all_episode_seeds[i : i + chunk_size]
        for i in range(0, n_episodes, chunk_size)
    ]

    def make_tasks(progress_queue: Any | None = None) -> list[_EvalWorkerTask]:
        config_dict = asdict(config)
        return [
            _EvalWorkerTask(
                env_id=env_id,
                episode_seeds=chunk,
                config_dict=config_dict,
                progress_queue=progress_queue,
            )
            for chunk in seed_chunks
        ]

    def update_progress(progress_queue: Any, progress_bar: tqdm, completed: int) -> int:
        while completed < n_episodes:
            try:
                progress_queue.get_nowait()
            except queue.Empty:
                break
            progress_bar.update(1)
            completed += 1
        return completed

    if workers == 1:
        results = [_run_eval_worker(make_tasks()[0])]
    else:
        if progress:
            manager = Manager()
            progress_bar = tqdm(
                total=n_episodes,
                desc=f"Eval ({workers} workers)",
                unit="episode",
            )
            try:
                progress_queue = manager.Queue()
                tasks = make_tasks(progress_queue)
                results_by_index: list[dict[str, list[int]] | None] = [
                    None
                    for _task in tasks
                ]
                completed = 0
                with ProcessPoolExecutor(max_workers=workers) as executor:
                    pending = {
                        executor.submit(_run_eval_worker, task): index
                        for index, task in enumerate(tasks)
                    }
                    while pending:
                        try:
                            progress_queue.get(timeout=0.1)
                        except queue.Empty:
                            pass
                        else:
                            progress_bar.update(1)
                            completed += 1
                        completed = update_progress(
                            progress_queue,
                            progress_bar,
                            completed,
                        )
                        finished = {future for future in pending if future.done()}
                        for future in finished:
                            index = pending.pop(future)
                            results_by_index[index] = future.result()
                results = [
                    result
                    for result in results_by_index
                    if result is not None
                ]
                completed = update_progress(progress_queue, progress_bar, completed)
            finally:
                progress_bar.close()
                manager.shutdown()
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                results = list(executor.map(_run_eval_worker, make_tasks()))

    all_lengths = []
    all_rewards = []
    all_max_tiles = []
    all_total_score = []
    all_illegal_counts = []
    for result in results:
        all_lengths.extend(result["lengths"])
        all_rewards.extend(result["rewards"])
        all_max_tiles.extend(result["max_tiles"])
        all_total_score.extend(result["total_score"])
        all_illegal_counts.extend(result["illegal_counts"])

    runtime = time.perf_counter() - start_time
    return all_lengths, all_rewards, all_max_tiles, all_total_score, all_illegal_counts, runtime


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

    if config.workers > 1:
        lengths, rewards, max_tiles, total_score, illegal_counts, runtime = run_episodes_parallel(
            env_id=config.env_id,
            config=config,
            n_episodes=config.episodes,
            seed=config.seed,
            workers=config.workers,
        )
    else:
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
    "run_episodes_parallel",
    "summarize_statistics",
]
