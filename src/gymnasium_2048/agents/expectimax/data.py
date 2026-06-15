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
from gymnasium_2048.agents.expectimax.search import evaluate_root_actions
from gymnasium_2048.agents.expectimax.symmetry import (
    NUM_SYMMETRIES,
    apply_symmetry,
    transform_action,
)
from gymnasium_2048.agents.heuristic.features import RewardTransform


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
    reward_transform: RewardTransform
    debug_fields: bool


def _empty_episode_payload(debug_fields: bool) -> dict[str, list]:
    payload: dict[str, list] = {
        "after_boards": [],
        "target_us": [],
        "immediate_rewards": [],
        "actions": [],
        "depths": [],
        "root_ids": [],
        "episodes": [],
        "steps": [],
        "scores_so_far": [],
        "max_tiles": [],
        "empty_counts": [],
    }
    if debug_fields:
        payload["root_boards"] = []
        payload["target_qs"] = []
    return payload


def _generate_episode_payload(task: EpisodeTask) -> dict[str, list]:
    policy = ExpectimaxPolicy(
        depth=task.depth,
        seed=task.policy_seed,
        chance_samples=task.chance_samples,
        full_chance_empty_threshold=task.full_chance_empty_threshold,
        reward_transform=task.reward_transform,
    )
    payload = _empty_episode_payload(task.debug_fields)

    env = gym.make(task.env_id)
    try:
        _observation, info = env.reset(seed=task.episode_seed)
        terminated = truncated = False
        step = 0

        while not terminated and not truncated:
            if task.max_steps is not None and step >= task.max_steps:
                break

            board = np.asarray(info["board"], dtype=np.uint8).copy()
            root_values = evaluate_root_actions(
                board,
                depth=task.depth,
                evaluator=policy.evaluator,
                reward_transform=task.reward_transform,
                rng=policy.rng,
                chance_samples=task.chance_samples,
                full_chance_empty_threshold=task.full_chance_empty_threshold,
            )
            root_id = task.episode * 1_000_000 + step

            for action in np.flatnonzero(root_values.legal_mask):
                action = int(action)
                payload["after_boards"].append(root_values.afterstates[action].copy())
                payload["target_us"].append(root_values.afterstate_values[action])
                payload["immediate_rewards"].append(
                    root_values.immediate_rewards[action]
                )
                payload["actions"].append(action)
                payload["depths"].append(task.depth)
                payload["root_ids"].append(root_id)
                payload["episodes"].append(task.episode)
                payload["steps"].append(step)
                payload["scores_so_far"].append(int(info["total_score"]))
                payload["max_tiles"].append(max_tile_value(board))
                payload["empty_counts"].append(count_empty(board))
                if task.debug_fields:
                    payload["root_boards"].append(board.copy())
                    payload["target_qs"].append(root_values.scores[action])

            action = (
                int(np.argmax(root_values.scores))
                if np.any(root_values.legal_mask)
                else 0
            )
            _observation, _reward, terminated, truncated, info = env.step(action)
            step += 1

        return payload
    finally:
        env.close()


def _merge_payloads(
    payloads: list[dict[str, list]],
    debug_fields: bool,
) -> dict[str, list]:
    merged = _empty_episode_payload(debug_fields)
    for payload in payloads:
        for key in merged:
            merged[key].extend(payload[key])
    return merged


def _board_array(values: list) -> np.ndarray:
    array = np.asarray(values, dtype=np.uint8)
    return array.reshape((-1, 4, 4))


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
    reward_transform: RewardTransform = "raw",
    debug_fields: bool = False,
    symmetry_augmentation: bool = False,
) -> dict[str, np.ndarray | dict[str, Any]]:
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    if depth < 0:
        raise ValueError("depth must be non-negative")
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
            reward_transform=reward_transform,
            debug_fields=debug_fields,
        )
        for episode in range(episodes)
    ]

    if workers == 1:
        iterator = (
            trange(episodes, desc="Generate", unit="episode")
            if progress
            else range(episodes)
        )
        payloads = [_generate_episode_payload(tasks[episode]) for episode in iterator]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            mapped = executor.map(_generate_episode_payload, tasks)
            if progress:
                mapped = tqdm(
                    mapped,
                    total=episodes,
                    desc="Generate",
                    unit="episode",
                )
            payloads = list(mapped)

    merged = _merge_payloads(payloads, debug_fields)
    root_ids = np.asarray(merged["root_ids"], dtype=np.int64)
    metadata = {
        "env_id": env_id,
        "episodes": episodes,
        "depth": depth,
        "seed": seed,
        "max_steps": max_steps,
        "chance_samples": chance_samples,
        "full_chance_empty_threshold": full_chance_empty_threshold,
        "workers": workers,
        "reward_transform": reward_transform,
        "episode_seeds": episode_seeds,
        "num_samples": len(merged["target_us"]),
        "num_roots": int(len(np.unique(root_ids))),
        "board_format": "exponent",
        "target": "afterstate_continuation_value",
        "action_order": ["up", "right", "down", "left"],
        "symmetry_augmentation": "none",
        "debug_fields": debug_fields,
    }

    dataset: dict[str, np.ndarray | dict[str, Any]] = {
        "after_boards": _board_array(merged["after_boards"]),
        "target_us": np.asarray(merged["target_us"], dtype=np.float32),
        "immediate_rewards": np.asarray(
            merged["immediate_rewards"],
            dtype=np.float32,
        ),
        "actions": np.asarray(merged["actions"], dtype=np.int64),
        "depths": np.asarray(merged["depths"], dtype=np.int64),
        "root_ids": root_ids,
        "episodes": np.asarray(merged["episodes"], dtype=np.int64),
        "steps": np.asarray(merged["steps"], dtype=np.int64),
        "scores_so_far": np.asarray(merged["scores_so_far"], dtype=np.int64),
        "max_tiles": np.asarray(merged["max_tiles"], dtype=np.int64),
        "empty_counts": np.asarray(merged["empty_counts"], dtype=np.int64),
        "metadata": metadata,
    }
    if debug_fields:
        dataset["root_boards"] = _board_array(merged["root_boards"])
        dataset["target_qs"] = np.asarray(merged["target_qs"], dtype=np.float32)
    return augment_afterstate_samples(dataset) if symmetry_augmentation else dataset


def augment_afterstate_samples(
    dataset: dict[str, np.ndarray | dict[str, Any]],
) -> dict[str, np.ndarray | dict[str, Any]]:
    """Expand each sample into all eight D4 symmetries."""
    boards = np.asarray(dataset["after_boards"], dtype=np.uint8)
    actions = np.asarray(dataset["actions"], dtype=np.int64)
    sample_count = len(boards)

    augmented: dict[str, np.ndarray | dict[str, Any]] = {}
    for key, value in dataset.items():
        if key == "metadata":
            continue
        array = np.asarray(value)
        if len(array) != sample_count:
            augmented[key] = array.copy()
            continue
        if key in {"after_boards", "root_boards"}:
            augmented[key] = np.stack(
                [
                    apply_symmetry(array[index], sym_id)
                    for index in range(sample_count)
                    for sym_id in range(NUM_SYMMETRIES)
                ]
            ).astype(np.uint8, copy=False)
        elif key == "actions":
            augmented[key] = np.asarray(
                [
                    transform_action(actions[index], sym_id)
                    for index in range(sample_count)
                    for sym_id in range(NUM_SYMMETRIES)
                ],
                dtype=np.int64,
            )
        else:
            augmented[key] = np.repeat(array, NUM_SYMMETRIES, axis=0)

    metadata = dict(dataset["metadata"])
    metadata["num_samples"] = sample_count * NUM_SYMMETRIES
    metadata["symmetry_augmentation"] = "all_8"
    metadata["source_num_samples"] = sample_count
    augmented["metadata"] = metadata
    return augmented


def _save_npz(
    dataset: dict[str, np.ndarray | dict[str, Any]],
    path: Path,
) -> None:
    arrays = {
        key: value
        for key, value in dataset.items()
        if isinstance(value, np.ndarray)
    }
    metadata = json.dumps(dataset["metadata"], sort_keys=True)
    np.savez_compressed(path, **arrays, metadata=np.asarray(metadata))


def save_expectimax_dataset(
    dataset: dict[str, np.ndarray | dict[str, Any]],
    path: str | Path,
    shard_size: int | None = None,
) -> list[Path]:
    output_path = Path(path)
    sample_count = len(np.asarray(dataset["target_us"]))
    if shard_size is None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _save_npz(dataset, output_path)
        return [output_path]
    if shard_size < 1:
        raise ValueError("shard_size must be positive when provided")

    output_path.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for shard_index, start in enumerate(range(0, sample_count, shard_size)):
        stop = min(start + shard_size, sample_count)
        shard: dict[str, np.ndarray | dict[str, Any]] = {}
        for key, value in dataset.items():
            if key == "metadata":
                continue
            array = np.asarray(value)
            shard[key] = array[start:stop]
        metadata = dict(dataset["metadata"])
        metadata.update(
            {
                "shard_index": shard_index,
                "shard_start": start,
                "shard_stop": stop,
                "num_samples": stop - start,
                "total_num_samples": sample_count,
            }
        )
        shard["metadata"] = metadata
        shard_path = output_path / f"dataset_part_{shard_index:03d}.npz"
        _save_npz(shard, shard_path)
        saved.append(shard_path)
    return saved


def _load_npz(path: Path) -> dict[str, np.ndarray | dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        dataset: dict[str, np.ndarray | dict[str, Any]] = {
            key: data[key].copy() for key in data.files if key != "metadata"
        }
        dataset["metadata"] = json.loads(str(data["metadata"].item()))
        return dataset


def load_expectimax_dataset(
    path: str | Path,
) -> dict[str, np.ndarray | dict[str, Any]]:
    input_path = Path(path)
    if input_path.is_file():
        return _load_npz(input_path)

    shard_paths = sorted(input_path.glob("dataset_part_*.npz"))
    if not shard_paths:
        raise FileNotFoundError(f"no dataset shards found in {input_path}")
    shards = [_load_npz(shard_path) for shard_path in shard_paths]
    keys = [key for key in shards[0] if key != "metadata"]
    dataset: dict[str, np.ndarray | dict[str, Any]] = {
        key: np.concatenate([np.asarray(shard[key]) for shard in shards], axis=0)
        for key in keys
    }
    metadata = dict(shards[0]["metadata"])
    metadata.pop("shard_index", None)
    metadata.pop("shard_start", None)
    metadata.pop("shard_stop", None)
    metadata["num_samples"] = len(np.asarray(dataset["target_us"]))
    metadata["shards"] = len(shards)
    dataset["metadata"] = metadata
    return dataset
