from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from tqdm import tqdm, trange

from gymnasium_2048.agents.expectimax.board import count_empty, max_tile_value
from gymnasium_2048.agents.expectimax.policy import ExpectimaxPolicy


@dataclass(frozen=True)
class EpisodeTask:
    env_id: str
    episode: int
    episode_seed: int
    depth: int
    policy_seed: int
    max_steps: int | None
    chance_samples: int | None
    full_chance_empty_threshold: int


def _empty_episode_payload() -> dict[str, list]:
    return {
        "boards": [],
        "legal_masks": [],
        "actions": [],
        "action_scores": [],
        "action_probs": [],
        "scores": [],
        "steps": [],
        "max_tiles": [],
        "empty_counts": [],
        "rewards": [],
        "episodes": [],
        "final_scores": [],
        "final_max_tiles": [],
        "depths": [],
    }


def _generate_episode_payload(task: EpisodeTask) -> dict[str, list]:
    policy = ExpectimaxPolicy(
        depth=task.depth,
        seed=task.policy_seed,
        chance_samples=task.chance_samples,
        full_chance_empty_threshold=task.full_chance_empty_threshold,
    )
    payload = _empty_episode_payload()

    env = gym.make(task.env_id)
    try:
        _observation, info = env.reset(seed=task.episode_seed)
        terminated = truncated = False
        step = 0
        episode_indices: list[int] = []

        while not terminated and not truncated:
            if task.max_steps is not None and step >= task.max_steps:
                break

            board = np.asarray(info["board"], dtype=np.uint8).copy()
            result = policy.analyze(board)

            payload["boards"].append(board)
            payload["legal_masks"].append(result.legal_mask.astype(bool))
            payload["actions"].append(result.action)
            payload["action_scores"].append(result.scores.astype(np.float32))
            payload["action_probs"].append(result.probabilities.astype(np.float32))
            payload["scores"].append(int(info["total_score"]))
            payload["steps"].append(step)
            payload["max_tiles"].append(max_tile_value(board))
            payload["empty_counts"].append(count_empty(board))
            payload["episodes"].append(task.episode)
            payload["depths"].append(task.depth)
            episode_indices.append(len(payload["boards"]) - 1)

            _observation, reward, terminated, truncated, info = env.step(result.action)
            payload["rewards"].append(float(reward))
            step += 1

        final_score = int(info["total_score"])
        final_max_tile = max_tile_value(np.asarray(info["board"], dtype=np.uint8))
        payload["final_scores"].extend([final_score] * len(episode_indices))
        payload["final_max_tiles"].extend([final_max_tile] * len(episode_indices))
        return payload
    finally:
        env.close()


def _merge_payloads(payloads: list[dict[str, list]]) -> dict[str, list]:
    merged = _empty_episode_payload()
    for payload in payloads:
        for key in merged:
            merged[key].extend(payload[key])
    return merged


def generate_expectimax_dataset(
    env_id: str = "gymnasium_2048:gymnasium_2048/TwentyFortyEight-v0",
    episodes: int = 10,
    depth: int = 1,
    seed: int = 42,
    max_steps: int | None = None,
    progress: bool = True,
    chance_samples: int | None = None,
    full_chance_empty_threshold: int = 6,
    workers: int = 1,
) -> dict[str, np.ndarray | dict[str, Any]]:
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    if workers < 1:
        raise ValueError("workers must be at least 1")

    rng = np.random.default_rng(seed)
    episode_seeds = rng.integers(0, 2**31 - 1, size=episodes).astype(int).tolist()
    tasks = [
        EpisodeTask(
            env_id=env_id,
            episode=episode,
            episode_seed=episode_seeds[episode],
            depth=depth,
            policy_seed=seed + episode * 1_000_003,
            max_steps=max_steps,
            chance_samples=chance_samples,
            full_chance_empty_threshold=full_chance_empty_threshold,
        )
        for episode in range(episodes)
    ]

    if workers == 1:
        iterator = trange(episodes, desc="Generate", unit="episode") if progress else range(episodes)
        payloads = [_generate_episode_payload(tasks[episode]) for episode in iterator]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            mapped = executor.map(_generate_episode_payload, tasks)
            if progress:
                mapped = tqdm(mapped, total=episodes, desc="Generate", unit="episode")
            payloads = list(mapped)

    merged = _merge_payloads(payloads)

    metadata = {
        "env_id": env_id,
        "episodes": episodes,
        "depth": depth,
        "seed": seed,
        "max_steps": max_steps,
        "chance_samples": chance_samples,
        "full_chance_empty_threshold": full_chance_empty_threshold,
        "workers": workers,
        "episode_seeds": episode_seeds,
        "num_samples": len(merged["boards"]),
        "board_format": "exponent",
        "action_order": ["up", "right", "down", "left"],
    }

    return {
        "boards": np.asarray(merged["boards"], dtype=np.uint8),
        "legal_masks": np.asarray(merged["legal_masks"], dtype=bool),
        "actions": np.asarray(merged["actions"], dtype=np.int64),
        "action_scores": np.asarray(merged["action_scores"], dtype=np.float32),
        "action_probs": np.asarray(merged["action_probs"], dtype=np.float32),
        "scores": np.asarray(merged["scores"], dtype=np.int64),
        "steps": np.asarray(merged["steps"], dtype=np.int64),
        "max_tiles": np.asarray(merged["max_tiles"], dtype=np.int64),
        "empty_counts": np.asarray(merged["empty_counts"], dtype=np.int64),
        "rewards": np.asarray(merged["rewards"], dtype=np.float32),
        "episodes": np.asarray(merged["episodes"], dtype=np.int64),
        "final_scores": np.asarray(merged["final_scores"], dtype=np.int64),
        "final_max_tiles": np.asarray(merged["final_max_tiles"], dtype=np.int64),
        "depths": np.asarray(merged["depths"], dtype=np.int64),
        "metadata": metadata,
    }


def save_expectimax_dataset(dataset: dict[str, np.ndarray | dict[str, Any]], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        key: value
        for key, value in dataset.items()
        if isinstance(value, np.ndarray)
    }
    metadata = json.dumps(dataset["metadata"], sort_keys=True)
    np.savez_compressed(output_path, **arrays, metadata=np.asarray(metadata))


def load_expectimax_dataset(path: str | Path) -> dict[str, np.ndarray | dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        dataset: dict[str, np.ndarray | dict[str, Any]] = {
            key: data[key].copy() for key in data.files if key != "metadata"
        }
        metadata_text = str(data["metadata"].item())
        dataset["metadata"] = json.loads(metadata_text)
        return dataset
