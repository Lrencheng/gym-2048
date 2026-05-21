from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
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
from gymnasium_2048.agents.ntuple.policy import (
    NTupleNetworkBasePolicy,
    NTupleNetworkQLearningPolicy,
    NTupleNetworkTDPolicy,
    NTupleNetworkTDPolicySmall,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NTupleTrainingConfig:
    agent: str = "tdl"
    env_id: str = DEFAULT_ENV_ID
    trained_agent: str = ""
    n_episodes: int = 100_000
    eval_freq: int = 5_000
    eval_episodes: int = 1000
    save_freq: int = -1
    seed: int = 42
    learning_rate: float = 0.0025
    out_dir: str = "models"


def make_ntuple_policy(
    agent: str,
    trained_agent: str = "",
) -> NTupleNetworkBasePolicy:
    policy_map: dict[str, type[NTupleNetworkBasePolicy]] = {
        "ql": NTupleNetworkQLearningPolicy,
        "tdl": NTupleNetworkTDPolicy,
        "tdl-small": NTupleNetworkTDPolicySmall,
    }
    policy_type = policy_map[agent]
    return policy_type.load(trained_agent) if trained_agent else policy_type()


def play_game(
    env: gym.Env,
    policy: NTupleNetworkBasePolicy,
    learn: bool = False,
    learning_rate: float = 0.0025,
) -> dict[str, Any]:
    _observation, info = env.reset()
    terminated = truncated = False

    while not terminated and not truncated:
        state = info["board"]
        action = policy.predict(state=state)
        _observation, reward, terminated, truncated, info = env.step(action)
        if learn:
            policy.learn(
                state=state,
                action=action,
                reward=float(reward),
                next_state=info["board"],
                learning_rate=learning_rate,
            )

    return info


def evaluate_ntuple_policy(
    env: gym.Env,
    policy: NTupleNetworkBasePolicy,
    eval_episodes: int,
) -> dict[str, Any]:
    winning_rate = 0
    total_score = 0
    max_tile = 0

    with trange(eval_episodes, desc="Evaluate", unit="episode", leave=False) as pbar:
        for _ in pbar:
            info = play_game(env=env, policy=policy)
            winning_rate += int(2 ** info["max"] >= 2048)
            total_score += info["total_score"]
            max_tile = max(max_tile, 2 ** info["max"])
            pbar.set_postfix({"max_tile": max_tile})

    return {
        "winning_rate": winning_rate / eval_episodes,
        "mean_score": total_score / eval_episodes,
        "max_tile": max_tile,
    }


def save_best_policy(out_dir: str, policy: NTupleNetworkBasePolicy) -> None:
    best_model_path = os.path.join(out_dir, "best_n_tuple_network_policy.zip")
    logger.info("new best model saved to %s", best_model_path)
    policy.save(path=best_model_path)


def save_checkpoint(
    episode: int,
    out_dir: str,
    policy: NTupleNetworkBasePolicy,
) -> None:
    checkpoint_path = os.path.join(out_dir, f"checkpoint_episode_{episode}.zip")
    logger.info("checkpoint saved to %s", checkpoint_path)
    policy.save(path=checkpoint_path)


def load_ntuple_training_config(
    agent: str,
    config_path: str | Path | None = None,
) -> tuple[NTupleTrainingConfig, dict[str, Any]]:
    source = Path(config_path) if config_path is not None else default_config_path(agent, "train")
    raw = validate_config_agent(load_yaml_mapping(source), agent, source)
    config = dataclass_from_mapping(NTupleTrainingConfig, raw, source)
    return config, asdict(config)


def train_ntuple(config: NTupleTrainingConfig) -> None:
    np.random.seed(config.seed)
    env = gym.make(config.env_id)
    policy = make_ntuple_policy(
        agent=config.agent,
        trained_agent=config.trained_agent,
    )
    best_mean_score = 0
    os.makedirs(config.out_dir, exist_ok=True)

    logger.info("start training n-tuple network")

    with trange(1, config.n_episodes + 1, desc="Train", unit="episode") as pbar:
        for episode in pbar:
            play_game(
                env=env,
                policy=policy,
                learn=True,
                learning_rate=config.learning_rate,
            )

            should_evaluate = config.eval_freq > 0 and (
                episode % config.eval_freq == 0 or episode == config.n_episodes
            )
            if should_evaluate:
                metrics = evaluate_ntuple_policy(
                    env=env,
                    policy=policy,
                    eval_episodes=config.eval_episodes,
                )
                logger.info(
                    "episode %d: winning rate = %.2f, mean score = %.2f, max tile = %d",
                    episode,
                    metrics["winning_rate"],
                    metrics["mean_score"],
                    metrics["max_tile"],
                )

                if metrics["mean_score"] > best_mean_score:
                    best_mean_score = metrics["mean_score"]
                    save_best_policy(out_dir=config.out_dir, policy=policy)

                metrics["best_mean_score"] = best_mean_score
                pbar.set_postfix(metrics)

            should_save = episode == config.n_episodes or (
                config.save_freq > 0 and episode % config.save_freq == 0
            )
            if should_save:
                save_checkpoint(episode=episode, out_dir=config.out_dir, policy=policy)

    env.close()
    logger.info("end training n-tuple network")


def train_ntuple_from_yaml(
    agent: str,
    config_path: str | Path | None = None,
    print_config: bool = False,
) -> NTupleTrainingConfig | None:
    config, printable = load_ntuple_training_config(
        agent=agent,
        config_path=config_path,
    )
    if print_config:
        print(dump_yaml_mapping(printable))
        return None

    train_ntuple(config)
    return config


__all__ = [
    "NTupleTrainingConfig",
    "evaluate_ntuple_policy",
    "load_ntuple_training_config",
    "make_ntuple_policy",
    "play_game",
    "train_ntuple",
    "train_ntuple_from_yaml",
]
