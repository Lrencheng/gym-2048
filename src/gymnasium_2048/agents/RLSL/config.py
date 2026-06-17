from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from gymnasium_2048.agents.config import (
    dataclass_from_mapping,
    default_config_path,
    dump_yaml_mapping,
    load_yaml_mapping,
    validate_config_agent,
)
from gymnasium_2048.agents.RLSL.train import RLSLTrainingConfig, train_rlsl

RLSL_RUN_ROOT = Path("models") / "RLSL"


@dataclass(frozen=True)
class RLSLTrainingYamlConfig:
    agent: str = "RLSL"
    out_root: str = str(RLSL_RUN_ROOT)
    run_name: str | None = None
    out_dir: str | None = None
    env_id: str = RLSLTrainingConfig.__dataclass_fields__["env_id"].default
    rounds: int = 20
    search_depth: int = 1
    num_eps_per_search: int = 1
    train_epochs_per_episode: int = 1
    batch_size: int = 8900
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-4
    l2_coeff: float | None = None
    loss_type: str = "mse"
    value_scale: float = 1.0
    target_mean: float | None = None
    target_std: float | None = None
    checkpoint_interval: int = 1
    eval_interval: int = 10
    eval_episodes: int = 10
    eval_workers: int = 1
    device: str = "auto"
    seed: int = 42
    num_workers: int = 0
    pin_memory: bool = True
    input_channels: int = 16
    conv_channels: int = 64
    hidden_size: int = 128
    chance_samples: int | None = None
    full_chance_empty_threshold: int = 6
    symmetry_augmentation: bool = True
    initial_checkpoint: str | None = None
    replay_enabled: bool = True
    replay_teacher_data_path: str | None = "data/expectimax_afterstates_400.npz"
    replay_capacity: int = 280_000
    replay_current_train_max_samples: int = 70_000
    replay_current_admission_fraction: float = 0.10
    replay_seed: int | None = None
    progress: bool = True


def resolve_rlsl_output_dir(
    out_root: str | Path = RLSL_RUN_ROOT,
    run_name: str | None = None,
    out_dir: str | Path | None = None,
) -> Path:
    root = Path(out_root)
    if out_dir:
        requested = Path(out_dir)
        return requested if requested.is_absolute() else root / requested.name
    if run_name:
        return root / run_name
    return root


def load_rlsl_training_config(
    config_path: str | Path | None = None,
    agent: str = "RLSL",
) -> tuple[RLSLTrainingConfig, dict[str, Any]]:
    source = Path(config_path) if config_path else default_config_path(agent, "train")
    raw = validate_config_agent(load_yaml_mapping(source), agent, source)
    yaml_config = dataclass_from_mapping(RLSLTrainingYamlConfig, raw, source)
    out_dir = resolve_rlsl_output_dir(
        yaml_config.out_root,
        yaml_config.run_name,
        yaml_config.out_dir,
    )
    values = asdict(yaml_config)
    training_values = {
        key: value
        for key, value in values.items()
        if key not in {"agent", "out_root", "run_name", "out_dir"}
    }
    config = RLSLTrainingConfig(out_dir=str(out_dir), **training_values)
    values["resolved_out_dir"] = str(out_dir)
    return config, values


def train_rlsl_from_yaml(
    config_path: str | Path | None = None,
    agent: str = "RLSL",
    print_config: bool = False,
) -> dict[str, Any] | None:
    config, printable = load_rlsl_training_config(config_path, agent)
    if print_config:
        print(dump_yaml_mapping(printable))
        return None
    result = train_rlsl(config)
    print(
        "RLSL search-improved value training complete: "
        f"device={result['device']}, "
        f"best_metric={result['best_metric']:.2f}, "
        f"best={result['best_checkpoint']}"
    )
    return result
