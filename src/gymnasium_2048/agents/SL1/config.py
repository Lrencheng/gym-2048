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
from gymnasium_2048.agents.SL1.train import (
    SupervisedTrainingConfig,
    train_supervised_cnn,
)


SUPERVISED_RUN_ROOT = Path("models") / "SL1"


@dataclass(frozen=True)
class SupervisedTrainingYamlConfig:
    agent: str = "SL1"
    data_path: str = "data/expectimax_d2_sampled_1000eps.npz"
    out_root: str = str(SUPERVISED_RUN_ROOT)
    run_name: str | None = None
    out_dir: str | None = None
    epochs: int = 10
    batch_size: int = 2048
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    temperature: float = 1.0
    validation_fraction: float = 0.2
    device: str = "cuda"
    seed: int = 42
    use_high_score_weighting: bool = True
    score_weight: float = 0.5
    tile_weight: float = 0.4
    late_game_weight: float = 0.2
    difficulty_weight: float = 0.2
    max_sample_weight: float = 3.0
    copy_training_data: bool = True
    num_workers: int = 4
    persistent_workers: bool = True
    prefetch_factor: int | None = 4
    pin_memory: bool = True
    drop_last: bool = False
    encode_on_device: bool = True
    compute_train_accuracy: bool = False
    validation_interval: int = 1
    validation_max_samples: int | None = 100_000
    amp: bool = True
    allow_tf32: bool = True
    torch_compile: bool = False
    profile: bool = True
    profile_batches: int = 50


def next_supervised_run_dir(root: Path | None = None, create: bool = True) -> Path:
    if root is None:
        root = SUPERVISED_RUN_ROOT
    if create:
        root.mkdir(parents=True, exist_ok=True)
    if not root.exists():
        return root / "train1"
    run_numbers = []
    for child in root.iterdir():
        if child.is_dir() and child.name.startswith("train"):
            suffix = child.name.removeprefix("train")
            if suffix.isdigit():
                run_numbers.append(int(suffix))
    return root / f"train{max(run_numbers, default=0) + 1}"


def resolve_supervised_output_dir(
    out_root: str | Path = SUPERVISED_RUN_ROOT,
    run_name: str | None = None,
    out_dir: str | Path | None = None,
    create_root: bool = True,
) -> Path:
    root = Path(out_root)
    if out_dir is None or str(out_dir).strip() == "":
        if run_name is None or run_name.strip() == "":
            return next_supervised_run_dir(root, create=create_root)
        return root / run_name

    requested = Path(out_dir)
    if requested.is_absolute():
        return requested

    normalized_parts = tuple(part.lower() for part in requested.parts)
    root_parts = tuple(part.lower() for part in root.parts)
    if normalized_parts in {("model",), ("models",), root_parts}:
        return next_supervised_run_dir(root, create=create_root)
    if normalized_parts[: len(root_parts)] == root_parts:
        return requested
    if requested.name and requested.name not in {".", ""}:
        return root / requested.name
    return next_supervised_run_dir(root, create=create_root)


def load_supervised_training_config(
    config_path: str | Path | None = None,
    agent: str = "SL1",
) -> tuple[SupervisedTrainingConfig, dict[str, Any]]:
    source = Path(config_path) if config_path is not None else default_config_path(agent, "train")
    raw = validate_config_agent(load_yaml_mapping(source), agent, source)
    yaml_config = dataclass_from_mapping(SupervisedTrainingYamlConfig, raw, source)
    out_dir = resolve_supervised_output_dir(
        out_root=yaml_config.out_root,
        run_name=yaml_config.run_name,
        out_dir=yaml_config.out_dir,
        create_root=False,
    )
    training_config = SupervisedTrainingConfig(
        data_path=yaml_config.data_path,
        out_dir=str(out_dir),
        epochs=yaml_config.epochs,
        batch_size=yaml_config.batch_size,
        learning_rate=yaml_config.learning_rate,
        weight_decay=yaml_config.weight_decay,
        temperature=yaml_config.temperature,
        validation_fraction=yaml_config.validation_fraction,
        device=yaml_config.device,
        seed=yaml_config.seed,
        use_high_score_weighting=yaml_config.use_high_score_weighting,
        score_weight=yaml_config.score_weight,
        tile_weight=yaml_config.tile_weight,
        late_game_weight=yaml_config.late_game_weight,
        difficulty_weight=yaml_config.difficulty_weight,
        max_sample_weight=yaml_config.max_sample_weight,
        copy_training_data=yaml_config.copy_training_data,
        num_workers=yaml_config.num_workers,
        persistent_workers=yaml_config.persistent_workers,
        prefetch_factor=yaml_config.prefetch_factor,
        pin_memory=yaml_config.pin_memory,
        drop_last=yaml_config.drop_last,
        encode_on_device=yaml_config.encode_on_device,
        compute_train_accuracy=yaml_config.compute_train_accuracy,
        validation_interval=yaml_config.validation_interval,
        validation_max_samples=yaml_config.validation_max_samples,
        amp=yaml_config.amp,
        allow_tf32=yaml_config.allow_tf32,
        torch_compile=yaml_config.torch_compile,
        profile=yaml_config.profile,
        profile_batches=yaml_config.profile_batches,
    )
    printable = asdict(yaml_config)
    printable["resolved_out_dir"] = str(out_dir)
    return training_config, printable


def train_supervised_from_yaml(
    config_path: str | Path | None = None,
    agent: str = "SL1",
    print_config: bool = False,
) -> dict[str, Any] | None:
    config, printable = load_supervised_training_config(
        config_path=config_path,
        agent=agent,
    )
    if print_config:
        print(dump_yaml_mapping(printable))
        return None

    result = train_supervised_cnn(config)
    print(
        "SL1 action-probability CNN training complete: "
        f"device={result['device']}, "
        f"output_dir={result['output_dir']}, "
        f"copied_data={result['copied_data_path']}, "
        f"best_validation_loss={result['best_validation_loss']:.6f}, "
        f"best={result['best_checkpoint']}, "
        f"last={result['last_checkpoint']}"
    )
    return result


__all__ = [
    "SUPERVISED_RUN_ROOT",
    "SupervisedTrainingYamlConfig",
    "load_supervised_training_config",
    "next_supervised_run_dir",
    "resolve_supervised_output_dir",
    "train_supervised_from_yaml",
]

