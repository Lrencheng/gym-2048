import argparse

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

from gymnasium_2048.agents.heuristic import HeuristicPolicy
from gymnasium_2048.agents.ntuple import (
    NTupleNetworkBasePolicy,
    NTupleNetworkQLearningPolicy,
    NTupleNetworkTDPolicy,
    NTupleNetworkTDPolicySmall,
)

plt.style.use("ggplot")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate 2048 agents",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--algo",
        help="RL Algorithm",
        choices=["ql", "tdl", "tdl-small", "heuristic"],
    )
    parser.add_argument(
        "--env",
        default="gymnasium_2048:gymnasium_2048/TwentyFortyEight-v0",
        help="environment id",
    )
    parser.add_argument(
        "-i",
        "--trained-agent",
        help="path to a trained agent",
    )
    parser.add_argument(
        "-n",
        "--n-episodes",
        type=int,
        default=1000,
        help="number of episodes",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random generator seed",
    )
    parser.add_argument(
        "-t",
        "--title",
        help="figure title",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        help="path to output png file",
    )
    args = parser.parse_args()
    return args


def make_env(env_id: str) -> gym.Env:
    env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env


def make_policy(
    algo: str,
    trained_agent: str | None = None,
) -> NTupleNetworkBasePolicy | HeuristicPolicy:
    """
    Makes the policy to evaluate.

    :param algo: Name of the algorithm.
    :param trained_agent: Path to a trained agent.
    :return: Policy.
    """
    if algo == "heuristic":
        return HeuristicPolicy()

    if trained_agent is None:
        raise ValueError(f"{algo} requires --trained-agent")

    algo_policy_map = {
        "ql": NTupleNetworkQLearningPolicy,
        "tdl": NTupleNetworkTDPolicy,
        "tdl-small": NTupleNetworkTDPolicySmall,
    }
    policy = algo_policy_map[algo]
    return policy.load(trained_agent)


def run_episodes(
    env: gym.Env,
    policy: NTupleNetworkBasePolicy | None,
    n_episodes: int,
) -> tuple[list[int], list[int], list[int], list[int]]:
    """
    Runs episodes and record statistics.

    :param env: Game environment.
    :param policy: Policy or None for random policy.
    :param n_episodes: Number of episodes.
    :return: Lengths, rewards, max tiles and total score.
    """
    lengths = []
    rewards = []
    max_tiles = []
    total_score = []

    for _ in trange(n_episodes, desc="Episode", unit="episode"):
        _observation, info = env.reset()
        terminated = truncated = False

        while not terminated and not truncated:
            if policy is None:
                action = env.action_space.sample()
            else:
                state = info["board"]
                action = policy.predict(state=state)

            _observation, _reward, terminated, truncated, info = env.step(action)

        lengths.append(info["episode"]["l"])
        rewards.append(info["episode"]["r"])
        max_tiles.append(info["max"])
        total_score.append(info["total_score"])

        env.reset()

    return lengths, rewards, max_tiles, total_score


def plot_statistics(
    lengths: list[int],
    rewards: list[int],
    max_tiles: list[int],
    total_score: list[int],
    title: str | None = None,
) -> plt.Figure:
    """
    Plots episode statistics.

    :param lengths: Lengths.
    :param rewards: Rewards.
    :param max_tiles: Maximum tiles reached.
    :param total_score: Total game score.
    :param title: Figure title. Default to None.
    :return: Figure with statistics.
    """
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
) -> dict[str, float | int]:
    """
    Summarizes the main evaluation metrics.

    :param lengths: Episode lengths.
    :param max_tiles: Maximum tiles reached as board exponents.
    :param total_score: Total game scores.
    :return: Summary metrics.
    """
    return {
        "episodes": len(total_score),
        "mean_score": float(np.mean(total_score)),
        "max_tile": int(np.max(np.power(2, max_tiles, dtype=int))),
        "mean_steps": float(np.mean(lengths)),
    }


def print_summary(summary: dict[str, float | int]) -> None:
    print(
        "Evaluation summary: "
        f"episodes={summary['episodes']}, "
        f"mean_score={summary['mean_score']:.2f}, "
        f"max_tile={summary['max_tile']}, "
        f"mean_steps={summary['mean_steps']:.2f}"
    )


def evaluate() -> None:
    args = parse_args()

    np.random.seed(args.seed)
    env = make_env(env_id=args.env)
    if args.algo is not None:
        policy = make_policy(algo=args.algo, trained_agent=args.trained_agent)
    else:
        policy = None

    lengths, rewards, max_tiles, total_score = run_episodes(
        env=env,
        policy=policy,
        n_episodes=args.n_episodes,
    )
    env.close()
    print_summary(
        summarize_statistics(
            lengths=lengths,
            max_tiles=max_tiles,
            total_score=total_score,
        )
    )
    fig = plot_statistics(
        lengths=lengths,
        rewards=rewards,
        max_tiles=max_tiles,
        total_score=total_score,
        title=args.title,
    )
    fig.show()

    if args.output_path is not None:
        fig.savefig(args.output_path)


if __name__ == "__main__":
    evaluate()
