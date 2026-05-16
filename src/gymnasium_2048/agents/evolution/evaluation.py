from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from gymnasium_2048.agents.heuristic import HeuristicPolicy, HeuristicWeights


@dataclass(frozen=True)
class EvaluationResult:
    fitness: float
    mean_score: float
    max_tile: int
    mean_steps: float


def make_episode_seeds(seed: int, episodes: int) -> list[int]:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2**31 - 1, size=episodes).astype(int).tolist()


def evaluate_weights(
    weights: HeuristicWeights,
    episode_seeds: Sequence[int],
    env_id: str = "gymnasium_2048:gymnasium_2048/TwentyFortyEight-v0",
) -> EvaluationResult:
    policy = HeuristicPolicy(weights=weights)
    scores = []
    max_tiles = []
    steps = []

    env = gym.make(env_id)
    try:
        for episode_seed in episode_seeds:
            _observation, info = env.reset(seed=int(episode_seed))
            terminated = truncated = False
            step_count = 0

            while not terminated and not truncated:
                action = policy.predict(state=info["board"])
                _observation, _reward, terminated, truncated, info = env.step(action)
                step_count += 1

            scores.append(info["total_score"])
            max_tiles.append(2 ** info["max"])
            steps.append(step_count)
    finally:
        env.close()

    mean_score = float(np.mean(scores))
    return EvaluationResult(
        fitness=mean_score,
        mean_score=mean_score,
        max_tile=int(np.max(max_tiles)),
        mean_steps=float(np.mean(steps)),
    )

