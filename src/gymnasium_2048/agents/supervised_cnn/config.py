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
from gymnasium_2048.agents.supervised_cnn.train import (
    SupervisedTrainingConfig,
    train_supervised_cnn,
)

SUPERVISED_RUN_ROOT = Path("models") / "supervised_cnn"


@dataclass(frozen=True)
class SupervisedTrainingYamlConfig:
    agent: str = "supervised_cnn"
    data_path: str = "data/expectimax_afterstates"
    out_root: str = str(SUPERVISED_RUN_ROOT)
    run_name: str | None = None
    out_dir: str | None = None
    epochs: int = 40
    batch_size: int = 2048
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    validation_fraction: float = 0.2
    device: str = "cuda"
    seed: int = 42
    num_workers: int = 4
    pin_memory: bool = True
    input_channels: int = 16
    conv_channels: int = 64
    hidden_size: int = 128
    loss: str = "huber"
    target_normalization: bool = True
    symmetry_augmentation: bool = True
    progress: bool = True


def next_supervised_run_dir(
    root: Path | None = None,
    create: bool = True,
) -> Path:
    root = root or SUPERVISED_RUN_ROOT
    if create:
        root.mkdir(parents=True, exist_ok=True)
    run_numbers = [
        int(child.name.removeprefix("train"))
        for child in root.iterdir()
        if child.is_dir()
        and child.name.startswith("train")
        and child.name.removeprefix("train").isdigit()
    ] if root.exists() else []
    return root / f"train{max(run_numbers, default=0) + 1}"


def resolve_supervised_output_dir(
    out_root: str | Path = SUPERVISED_RUN_ROOT,
    run_name: str | None = None,
    out_dir: str | Path | None = None,
    create_root: bool = True,
) -> Path:
    root = Path(out_root)
    if out_dir:
        requested = Path(out_dir)
        return requested if requested.is_absolute() else root / requested.name
    if run_name:
        return root / run_name
    return next_supervised_run_dir(root, create=create_root)


def load_supervised_training_config(
    config_path: str | Path | None = None,
    agent: str = "supervised_cnn",
) -> tuple[SupervisedTrainingConfig, dict[str, Any]]:
    source = Path(config_path) if config_path else default_config_path(agent, "train")
    raw = validate_config_agent(load_yaml_mapping(source), agent, source)
    yaml_config = dataclass_from_mapping(SupervisedTrainingYamlConfig, raw, source)
    out_dir = resolve_supervised_output_dir(
        yaml_config.out_root,
        yaml_config.run_name,
        yaml_config.out_dir,
        create_root=False,
    )
    values = asdict(yaml_config)
    training_values = {
        key: value
        for key, value in values.items()
        if key not in {"agent", "out_root", "run_name", "out_dir"}
    }
    config = SupervisedTrainingConfig(
        out_dir=str(out_dir),
        **training_values,
    )
    values["resolved_out_dir"] = str(out_dir)
    return config, values


def train_supervised_from_yaml(
    config_path: str | Path | None = None,
    agent: str = "supervised_cnn",
    print_config: bool = False,
) -> dict[str, Any] | None:
    config, printable = load_supervised_training_config(config_path, agent)
    if print_config:
        print(dump_yaml_mapping(printable))
        return None
    result = train_supervised_cnn(config)
    print(
        "Supervised CNN value training complete: "
        f"device={result['device']}, "
        f"best_validation_loss={result['best_validation_loss']:.6f}, "
        f"best={result['best_checkpoint']}"
    )
    return result
