from __future__ import annotations

from pathlib import Path
from typing import Any

from gymnasium_2048.agents.config import resolve_agent


TRAIN_AGENTS = {
    "ql",
    "tdl",
    "tdl-small",
    "supervised_cnn",
    "supervised_ntuple",
}
EVALUATE_AGENTS = {
    "ql",
    "tdl",
    "tdl-small",
    "heuristic",
    "expectimax",
    "supervised_cnn",
    "supervised_ntuple",
}


def run_train_command(
    agent: str | None,
    config_path: str | Path | None,
    print_config: bool = False,
) -> Any:
    resolved_agent = resolve_agent(agent, config_path, TRAIN_AGENTS)
    if resolved_agent == "supervised_cnn":
        from gymnasium_2048.agents.supervised_cnn.config import train_supervised_from_yaml

        return train_supervised_from_yaml(
            config_path=config_path,
            agent=resolved_agent,
            print_config=print_config,
        )
    if resolved_agent == "supervised_ntuple":
        from gymnasium_2048.agents.supervised_ntuple.config import (
            train_supervised_ntuple_from_yaml,
        )

        return train_supervised_ntuple_from_yaml(
            config_path=config_path,
            agent=resolved_agent,
            print_config=print_config,
        )
    from gymnasium_2048.agents.ntuple.training import train_ntuple_from_yaml

    return train_ntuple_from_yaml(
        agent=resolved_agent,
        config_path=config_path,
        print_config=print_config,
    )


def run_evaluate_command(
    agent: str | None,
    config_path: str | Path | None,
    print_config: bool = False,
) -> Any:
    resolved_agent = resolve_agent(agent, config_path, EVALUATE_AGENTS)
    from gymnasium_2048.agents.evaluation import evaluate_from_yaml

    return evaluate_from_yaml(
        agent=resolved_agent,
        config_path=config_path,
        print_config=print_config,
    )
