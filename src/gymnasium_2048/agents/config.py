from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml


DEFAULT_ENV_ID = "gymnasium_2048:gymnasium_2048/TwentyFortyEight-v0"
REPO_ROOT = Path(__file__).resolve().parents[3]


class ConfigError(ValueError):
    """Raised when an agent YAML config is missing or malformed."""


T = TypeVar("T")


def load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"config file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, dict):
        raise ConfigError(f"config file must contain a YAML mapping: {config_path}")
    return dict(data)


def dump_yaml_mapping(data: dict[str, Any]) -> str:
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


def validate_config_agent(
    config: dict[str, Any],
    expected_agent: str,
    source: str | Path,
) -> dict[str, Any]:
    actual_agent = config.get("agent")
    if actual_agent is not None and actual_agent != expected_agent:
        raise ConfigError(
            f"{source} declares agent={actual_agent!r}, "
            f"but command requested agent={expected_agent!r}"
        )
    config = dict(config)
    config["agent"] = expected_agent
    return config


def dataclass_from_mapping(
    config_type: type[T],
    data: dict[str, Any],
    source: str | Path,
    ignored_keys: set[str] | None = None,
) -> T:
    if not is_dataclass(config_type):
        raise TypeError(f"{config_type!r} is not a dataclass type")

    ignored = ignored_keys or set()
    allowed = {field.name for field in fields(config_type)}
    unknown = sorted(set(data) - allowed - ignored)
    if unknown:
        names = ", ".join(unknown)
        raise ConfigError(f"unknown config field(s) in {source}: {names}")

    kwargs = {key: data[key] for key in allowed if key in data}
    return config_type(**kwargs)


def default_config_path(agent: str, task: str) -> Path:
    if agent in {"ql", "tdl", "tdl-small"}:
        return Path(__file__).resolve().parent / "ntuple" / "configs" / f"{task}_{agent}.yaml"
    return Path(__file__).resolve().parent / agent / "configs" / f"{task}.yaml"


def infer_agent_from_config(path: str | Path) -> str:
    config = load_yaml_mapping(path)
    agent = config.get("agent")
    if not isinstance(agent, str) or not agent:
        raise ConfigError(f"{path} must define a non-empty 'agent' field")
    return agent


def resolve_agent(
    agent: str | None,
    config_path: str | Path | None,
    valid_agents: set[str],
) -> str:
    resolved = agent
    if resolved is None and config_path is not None:
        resolved = infer_agent_from_config(config_path)
    if resolved is None:
        raise ConfigError("--agent is required when --config is not provided")
    if resolved not in valid_agents:
        allowed = ", ".join(sorted(valid_agents))
        raise ConfigError(f"unknown agent {resolved!r}; expected one of: {allowed}")
    return resolved
