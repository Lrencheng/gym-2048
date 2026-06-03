from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy as np

from gymnasium_2048.agents.config import ConfigError
from gymnasium_2048.agents.expectimax import ExpectimaxHeuristicWeights, ExpectimaxPolicy
from gymnasium_2048.agents.heuristic import HeuristicPolicy, HeuristicWeights

WeightFactory = Callable[..., Any]


@dataclass(frozen=True)
class AgentSpec:
    agent: str
    weight_type: WeightFactory
    parameter_names: tuple[str, ...]
    parameter_bounds: np.ndarray
    policy_config: dict[str, Any]

    def clip_vector(self, vector: np.ndarray) -> np.ndarray:
        return np.clip(vector, self.parameter_bounds[:, 0], self.parameter_bounds[:, 1])

    def vector_to_weights(self, vector: np.ndarray) -> Any:
        clipped = self.clip_vector(np.asarray(vector, dtype=np.float64))
        values = dict(zip(self.parameter_names, clipped.tolist()))
        return self.weight_type(**values)

    def weights_to_vector(self, weights: Any) -> np.ndarray:
        data = asdict(weights)
        return np.array([data[name] for name in self.parameter_names], dtype=np.float64)

    def weights_to_dict(self, weights: Any) -> dict[str, float]:
        return dict(zip(self.parameter_names, self.weights_to_vector(weights).tolist()))

    def make_policy(self, weights: Any) -> Any:
        reward_transform = self.policy_config.get("reward_transform")
        if self.agent == "heuristic":
            return HeuristicPolicy(
                weights=weights,
                reward_transform=reward_transform or "raw",
            )
        if self.agent == "expectimax":
            return ExpectimaxPolicy(
                depth=int(self.policy_config.get("depth", 2)),
                temperature=float(self.policy_config.get("temperature", 1.0)),
                weights=weights,
                reward_transform=reward_transform or "log2p1",
                seed=self.policy_config.get("seed"),
                chance_samples=self.policy_config.get("chance_samples"),
                full_chance_empty_threshold=int(
                    self.policy_config.get("full_chance_empty_threshold", 6)
                ),
            )
        raise ValueError(f"unsupported evolution agent: {self.agent!r}")


def _weight_fields(weight_type: WeightFactory) -> tuple[str, ...]:
    return tuple(field.name for field in fields(weight_type))


def _bounds_array(
    weight_type: WeightFactory,
    bounds: dict[str, tuple[float, float]],
    source: str,
) -> tuple[tuple[str, ...], np.ndarray]:
    names = _weight_fields(weight_type)
    expected = set(names)
    actual = set(bounds)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if extra:
            details.append(f"extra: {', '.join(extra)}")
        raise ConfigError(f"parameter_bounds mismatch for {source}: {'; '.join(details)}")

    return names, np.array([bounds[name] for name in names], dtype=np.float64)


def make_agent_spec(
    agent: str,
    parameter_bounds: dict[str, tuple[float, float]],
    policy_config: dict[str, Any] | None = None,
) -> AgentSpec:
    if agent == "heuristic":
        weight_type = HeuristicWeights
    elif agent == "expectimax":
        weight_type = ExpectimaxHeuristicWeights
    else:
        raise ConfigError(f"unsupported evolution agent: {agent!r}")

    names, bounds = _bounds_array(weight_type, parameter_bounds, agent)
    return AgentSpec(
        agent=agent,
        weight_type=weight_type,
        parameter_names=names,
        parameter_bounds=bounds,
        policy_config=dict(policy_config or {}),
    )
