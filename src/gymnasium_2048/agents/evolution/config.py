from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gymnasium_2048.agents.config import (
    ConfigError,
    default_config_path,
    load_yaml_mapping,
    validate_config_agent,
)


DEFAULT_SEED = 42
EVOLUTION_AGENTS = {"heuristic", "expectimax"}

PARAMETER_NAMES = (
    "empty_cells",
    "smoothness",
    "monotonicity",
    "corner_max",
    "merge_potential",
    "edge_bonus",
    "reward",
)

PARAMETER_BOUNDS = np.array(
    [
        (5.0, 20.0),
        (0.5, 3.0),
        (1.0, 5.0),
        (2.0, 5.0),
        (2.0, 8.0),
        (0.01, 0.1),
        (0.5, 2.0),
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class EvolutionConfig:
    agent: str = "heuristic"
    env_id: str = "gymnasium_2048:gymnasium_2048/TwentyFortyEight-v0"
    out_dir: str = "models/evolution"
    seed: int = DEFAULT_SEED
    population_size: int = 8
    generations: int = 5
    episodes_per_candidate: int = 5
    workers: int = 1
    elite_size: int = 2
    tournament_size: int = 3
    crossover_rate: float = 0.8
    mutation_rate: float = 0.25
    mutation_scale: float = 0.15

    def validate(self) -> None:
        if self.agent not in EVOLUTION_AGENTS:
            allowed = ", ".join(sorted(EVOLUTION_AGENTS))
            raise ValueError(f"agent must be one of: {allowed}")
        if not isinstance(self.out_dir, str) or not self.out_dir.strip():
            raise ValueError("out_dir must not be empty")
        if self.population_size < 2:
            raise ValueError("population_size must be at least 2")
        if self.generations < 1:
            raise ValueError("generations must be at least 1")
        if self.episodes_per_candidate < 1:
            raise ValueError("episodes_per_candidate must be at least 1")
        if self.workers < 1:
            raise ValueError("workers must be at least 1")
        if not 1 <= self.elite_size < self.population_size:
            raise ValueError("elite_size must be in [1, population_size)")
        if not 1 <= self.tournament_size <= self.population_size:
            raise ValueError("tournament_size must be in [1, population_size]")
        if not 0.0 <= self.crossover_rate <= 1.0:
            raise ValueError("crossover_rate must be in [0, 1]")
        if not 0.0 <= self.mutation_rate <= 1.0:
            raise ValueError("mutation_rate must be in [0, 1]")
        if self.mutation_scale < 0.0:
            raise ValueError("mutation_scale must be non-negative")


def default_train_config_path(agent: str) -> Path:
    if agent not in EVOLUTION_AGENTS:
        allowed = ", ".join(sorted(EVOLUTION_AGENTS))
        raise ConfigError(f"unknown evolution agent {agent!r}; expected one of: {allowed}")
    return Path(__file__).resolve().parent / "configs" / f"train_{agent}.yaml"


def resolve_relative_config_path(source: str | Path, path: str | Path) -> Path:
    config_path = Path(path)
    if config_path.is_absolute():
        return config_path
    return (Path(source).resolve().parent / config_path).resolve()


def _coerce_parameter_bounds(data: object, source: str | Path) -> dict[str, tuple[float, float]]:
    if not isinstance(data, dict):
        raise ConfigError(f"parameter_bounds must be a mapping in {source}")

    bounds = {}
    for name, raw_range in data.items():
        if (
            not isinstance(name, str)
            or not isinstance(raw_range, (list, tuple))
            or len(raw_range) != 2
        ):
            raise ConfigError(
                f"parameter_bounds entries must be name: [lower, upper] in {source}"
            )
        lower = float(raw_range[0])
        upper = float(raw_range[1])
        if lower > upper:
            raise ConfigError(f"parameter bound lower > upper for {name!r} in {source}")
        bounds[name] = (lower, upper)
    return bounds


def load_evolution_config(
    agent: str = "heuristic",
    config_path: str | Path | None = None,
) -> tuple[EvolutionConfig, dict[str, Any], dict[str, tuple[float, float]]]:
    source = Path(config_path) if config_path is not None else default_train_config_path(agent)
    raw = validate_config_agent(load_yaml_mapping(source), agent, source)
    bounds = _coerce_parameter_bounds(raw.pop("parameter_bounds", None), source)

    policy_config_path = raw.pop("policy_config", None)
    policy_config: dict[str, Any] = {}
    if policy_config_path is not None:
        resolved = resolve_relative_config_path(source, policy_config_path)
        policy_config = validate_config_agent(load_yaml_mapping(resolved), agent, resolved)
    else:
        policy_config = validate_config_agent(
            load_yaml_mapping(default_config_path(agent, "evaluate")),
            agent,
            default_config_path(agent, "evaluate"),
        )

    policy_override_keys = {
        "depth",
        "temperature",
        "chance_samples",
        "full_chance_empty_threshold",
        "weights",
        "reward_transform",
    }
    policy_overrides = {
        key: raw.pop(key) for key in list(raw) if key in policy_override_keys
    }
    merged_policy_config = {**policy_config, **policy_overrides}

    allowed = {field.name for field in EvolutionConfig.__dataclass_fields__.values()}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        names = ", ".join(unknown)
        raise ConfigError(f"unknown evolution config field(s) in {source}: {names}")

    config_data = {key: raw[key] for key in allowed if key in raw}
    config_data["agent"] = agent
    if "env_id" not in config_data and "env_id" in merged_policy_config:
        config_data["env_id"] = merged_policy_config["env_id"]

    return EvolutionConfig(**config_data), merged_policy_config, bounds
