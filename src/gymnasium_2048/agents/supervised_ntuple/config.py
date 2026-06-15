from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from gymnasium_2048.agents.config import (
    dataclass_from_mapping,
    default_config_path,
    dump_yaml_mapping,
    load_yaml_mapping,
    validate_config_agent,
)
from gymnasium_2048.agents.supervised_ntuple.train import (
    SupervisedNTupleTrainingConfig,
    train_supervised_ntuple,
)


@dataclass(frozen=True)
class SupervisedNTupleYamlConfig:
    agent: str = "supervised_ntuple"
    data_path: str = "data/expectimax_afterstates"
    model_path: str = "models/supervised_ntuple/value_model.npz"
    pattern_set: str = "rows_cols"
    custom_patterns: Sequence[Sequence[Sequence[int]]] | None = None
    num_values: int = 16
    learning_rate: float = 0.05
    epochs: int = 20
    shuffle: bool = True
    weight_decay: float = 0.0
    target_normalization: bool = True
    validation_fraction: float = 0.2
    seed: int = 42


def load_supervised_ntuple_config(
    config_path: str | Path | None = None,
    agent: str = "supervised_ntuple",
) -> tuple[SupervisedNTupleTrainingConfig, dict[str, Any]]:
    source = Path(config_path) if config_path else default_config_path(agent, "train")
    raw = validate_config_agent(load_yaml_mapping(source), agent, source)
    yaml_config = dataclass_from_mapping(
        SupervisedNTupleYamlConfig,
        raw,
        source,
    )
    values = asdict(yaml_config)
    values.pop("agent")
    return SupervisedNTupleTrainingConfig(**values), asdict(yaml_config)


def train_supervised_ntuple_from_yaml(
    config_path: str | Path | None = None,
    agent: str = "supervised_ntuple",
    print_config: bool = False,
) -> dict[str, Any] | None:
    config, printable = load_supervised_ntuple_config(config_path, agent)
    if print_config:
        print(dump_yaml_mapping(printable))
        return None
    result = train_supervised_ntuple(config)
    print(
        "Supervised n-tuple value training complete: "
        f"model={result['model_path']}, "
        f"train_loss={result['history'][-1]['train_loss']:.6f}"
    )
    return result
