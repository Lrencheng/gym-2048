import argparse
import time
from collections import Counter
from typing import Protocol

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

from gymnasium_2048.agents.expectimax import ExpectimaxPolicy
from gymnasium_2048.agents.heuristic import HeuristicPolicy
from gymnasium_2048.agents.ntuple import (
    NTupleNetworkBasePolicy,
    NTupleNetworkQLearningPolicy,
    NTupleNetworkTDPolicy,
    NTupleNetworkTDPolicySmall,
)
from gymnasium_2048.agents.supervised_cnn import SupervisedCNNPolicy

plt.style.use("ggplot")


class PredictPolicy(Protocol):
    def predict(self, state: np.ndarray) -> int:
        ...


AGENT_CHOICES = [
    "ql",
    "tdl",
    "tdl-small",
    "heuristic",
    "expectimax",
    "supervised_cnn",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate 2048 agents",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--algo",
        "--agent",
        dest="algo",
        help="agent or RL algorithm",
        choices=AGENT_CHOICES,
    )
    parser.add_argument(
        "--env",
        default="gymnasium_2048:gymnasium_2048/TwentyFortyEight-v0",
        help="environment id",
    )
    parser.add_argument(
        "-i",
        "--trained-agent",
        "--checkpoint",
        dest="trained_agent",
        help="path to a trained agent or checkpoint",
    )
    parser.add_argument(
        "-n",
        "--n-episodes",
        "--episodes",
        dest="n_episodes",
        type=int,
        default=1000,
        help="number of episodes",
    )
    parser.add_argument("--depth", type=int, default=2, help="Expectimax search depth")
    parser.add_argument(
        "--chance-samples",
        type=int,
        help="sample empty cells at Expectimax chance nodes above the full threshold",
    )
    parser.add_argument(
        "--full-chance-empty-threshold",
        type=int,
        default=6,
        help="empty-cell count at or below which Expectimax chance nodes enumerate all cells",
    )
    parser.add_argument("--device", default="cpu", help="device for supervised CNN")
    parser.add_argument("--seed", type=int, default=42, help="random generator seed")
    parser.add_argument("-t", "--title", help="figure title")
    parser.add_argument("-o", "--output-path", help="path to output png file")
    parser.add_argument("--no-plot", action="store_true", help="skip interactive plot")
    return parser.parse_args()


def make_env(env_id: str) -> gym.Env:
    env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env


def make_policy(
    algo: str,
    trained_agent: str | None = None,
    depth: int = 2,
    seed: int | None = None,
    device: str = "cpu",
    chance_samples: int | None = None,
    full_chance_empty_threshold: int = 6,
) -> PredictPolicy | None:
    if algo == "heuristic":
        return HeuristicPolicy()
    if algo == "expectimax":
        return ExpectimaxPolicy(
            depth=depth,
            seed=seed,
            chance_samples=chance_samples,
            full_chance_empty_threshold=full_chance_empty_threshold,
        )
    if algo == "supervised_cnn":
        if trained_agent is None:
            raise ValueError("supervised_cnn requires --checkpoint")
        return SupervisedCNNPolicy.load(trained_agent, device=device, seed=seed)
    if algo is None:
        return None

    if trained_agent is None:
        raise ValueError(f"{algo} requires --trained-agent")

    algo_policy_map: dict[str, type[NTupleNetworkBasePolicy]] = {
        "ql": NTupleNetworkQLearningPolicy,
        "tdl": NTupleNetworkTDPolicy,
        "tdl-small": NTupleNetworkTDPolicySmall,
    }
    policy = algo_policy_map[algo]
    return policy.load(trained_agent)


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
            if policy is None:
                action = env.action_space.sample()
            else:
                action = policy.predict(state=info["board"])

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


def evaluate() -> None:
    args = parse_args()

    np.random.seed(args.seed)
    env = make_env(env_id=args.env)
    policy = (
        make_policy(
            algo=args.algo,
            trained_agent=args.trained_agent,
            depth=args.depth,
            seed=args.seed,
            device=args.device,
            chance_samples=args.chance_samples,
            full_chance_empty_threshold=args.full_chance_empty_threshold,
        )
        if args.algo is not None
        else None
    )

    lengths, rewards, max_tiles, total_score, illegal_counts, runtime = run_episodes(
        env=env,
        policy=policy,
        n_episodes=args.n_episodes,
        seed=args.seed,
    )
    env.close()
    print_summary(
        summarize_statistics(
            lengths=lengths,
            max_tiles=max_tiles,
            total_score=total_score,
            illegal_counts=illegal_counts,
            runtime=runtime,
        )
    )

    if not args.no_plot or args.output_path is not None:
        fig = plot_statistics(
            lengths=lengths,
            rewards=rewards,
            max_tiles=max_tiles,
            total_score=total_score,
            title=args.title,
        )
        if args.output_path is not None:
            fig.savefig(args.output_path)
        if not args.no_plot:
            fig.show()


if __name__ == "__main__":
    evaluate()
