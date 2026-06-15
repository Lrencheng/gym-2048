import argparse
from typing import Any, Protocol

import gymnasium as gym
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
from gymnasium_2048.agents.supervised_ntuple import SupervisedNTuplePolicy


class PredictPolicy(Protocol):
    def predict(self, state: np.ndarray) -> int:
        ...


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enjoy a 2048 trained agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--algo",
        "--agent",
        dest="algo",
        default="tdl",
        help="agent or RL algorithm",
        choices=[
            "ql",
            "tdl",
            "tdl-small",
            "heuristic",
            "expectimax",
            "supervised_cnn",
            "supervised_ntuple",
        ],
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
        default="",
        help="path to a trained agent or checkpoint",
    )
    parser.add_argument(
        "-n",
        "--n-episodes",
        type=int,
        default=1,
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
        help=(
            "empty-cell count at or below which Expectimax chance nodes "
            "enumerate all cells"
        ),
    )
    parser.add_argument("--device", default="cpu", help="device for supervised CNN")
    parser.add_argument("--seed", type=int, default=42, help="random generator seed")
    parser.add_argument("--record-video", action="store_true", help="record videos")
    parser.add_argument(
        "--video-folder",
        default="videos",
        help="path to videos folder",
    )
    return parser.parse_args()


def make_policy(
    algo: str,
    trained_agent: str = "",
    depth: int = 2,
    seed: int | None = None,
    device: str = "cpu",
    chance_samples: int | None = None,
    full_chance_empty_threshold: int = 6,
) -> PredictPolicy:
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
        if not trained_agent:
            raise ValueError("supervised_cnn requires --checkpoint")
        return SupervisedCNNPolicy.load(
            trained_agent,
            depth=depth,
            device=device,
            seed=seed,
            chance_samples=chance_samples,
            full_chance_empty_threshold=full_chance_empty_threshold,
        )
    if algo == "supervised_ntuple":
        if not trained_agent:
            raise ValueError("supervised_ntuple requires --trained-agent")
        return SupervisedNTuplePolicy.load(
            trained_agent,
            depth=depth,
            seed=seed,
            chance_samples=chance_samples,
            full_chance_empty_threshold=full_chance_empty_threshold,
        )

    algo_policy_map: dict[str, type[NTupleNetworkBasePolicy]] = {
        "ql": NTupleNetworkQLearningPolicy,
        "tdl": NTupleNetworkTDPolicy,
        "tdl-small": NTupleNetworkTDPolicySmall,
    }
    policy = algo_policy_map[algo]
    return policy.load(trained_agent)


def play_game(env: gym.Env, policy: PredictPolicy) -> dict[str, Any]:
    _observation, info = env.reset()
    terminated = truncated = False

    while not terminated and not truncated:
        state = info["board"]
        action = policy.predict(state=state)
        _observation, _reward, terminated, truncated, info = env.step(action)

    return info


def enjoy() -> None:
    args = parse_args()

    np.random.seed(args.seed)
    if args.record_video:
        env = gym.make(args.env, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=args.video_folder,
            episode_trigger=lambda _: True,
            disable_logger=True,
        )
    else:
        env = gym.make(args.env, render_mode="human")

    policy = make_policy(
        algo=args.algo,
        trained_agent=args.trained_agent,
        depth=args.depth,
        seed=args.seed,
        device=args.device,
        chance_samples=args.chance_samples,
        full_chance_empty_threshold=args.full_chance_empty_threshold,
    )

    for _ in trange(args.n_episodes, desc="Enjoy"):
        play_game(env=env, policy=policy)

    env.close()


if __name__ == "__main__":
    enjoy()
