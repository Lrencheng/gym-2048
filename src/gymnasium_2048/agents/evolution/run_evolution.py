from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from gymnasium_2048.agents.evolution.config import (
    EVOLUTION_AGENTS,
    EvolutionConfig,
    load_evolution_config,
)
from gymnasium_2048.agents.evolution.genetic import (
    EvolutionResult,
    GeneticOptimizer,
)
from gymnasium_2048.agents.evolution.plotting import plot_history
from gymnasium_2048.agents.evolution.specs import make_agent_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize 2048 heuristic weights with a genetic algorithm",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--agent",
        choices=sorted(EVOLUTION_AGENTS),
        default="heuristic",
        help="agent weight schema to optimize",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="evolution training YAML; defaults to evolution/configs/train_<agent>.yaml",
    )
    parser.add_argument(
        "--env",
        default=None,
        help="environment id",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--population-size", type=int, default=None)
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--elite-size", type=int, default=None)
    parser.add_argument("--tournament-size", type=int, default=None)
    parser.add_argument("--crossover-rate", type=float, default=None)
    parser.add_argument("--mutation-rate", type=float, default=None)
    parser.add_argument("--mutation-scale", type=float, default=None)
    parser.add_argument(
        "--out-dir",
        default=None,
        help="directory for best parameters, history csv and plot; overrides train YAML",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="disable the terminal progress bar",
    )
    return parser.parse_args()


def make_config(
    args: argparse.Namespace,
) -> tuple[EvolutionConfig, dict[str, object], dict[str, tuple[float, float]]]:
    config, policy_config, bounds = load_evolution_config(
        agent=args.agent,
        config_path=args.config,
    )
    overrides = {
        "env_id": args.env,
        "seed": args.seed,
        "population_size": args.population_size,
        "generations": args.generations,
        "episodes_per_candidate": args.episodes,
        "workers": args.workers,
        "elite_size": args.elite_size,
        "tournament_size": args.tournament_size,
        "crossover_rate": args.crossover_rate,
        "mutation_rate": args.mutation_rate,
        "mutation_scale": args.mutation_scale,
        "out_dir": args.out_dir,
    }
    data = asdict(config)
    data.update({key: value for key, value in overrides.items() if value is not None})
    return EvolutionConfig(**data), policy_config, bounds


def save_best_parameters(
    result: EvolutionResult,
    config: EvolutionConfig,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    weights = (result.train_config or {}).get("best_weights", {})
    payload = {
        "agent": result.agent,
        "seed": config.seed,
        "config": asdict(config),
        "policy_config": result.policy_config,
        "train_config": result.train_config,
        "fitness": result.best_evaluation.fitness,
        "mean_score": result.best_evaluation.mean_score,
        "max_tile": result.best_evaluation.max_tile,
        "mean_steps": result.best_evaluation.mean_steps,
        "weights": weights,
    }
    output_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def save_history(result: EvolutionResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "generation",
                "best_fitness",
                "mean_fitness",
                "best_mean_score",
                "best_max_tile",
                "best_mean_steps",
                "best_vector",
            ]
        )
        for record in result.history:
            writer.writerow(
                [
                    record.generation,
                    record.best_fitness,
                    record.mean_fitness,
                    record.best_mean_score,
                    record.best_max_tile,
                    record.best_mean_steps,
                    " ".join(f"{value:.8f}" for value in record.best_vector),
                ]
            )


def print_result(result: EvolutionResult, out_dir: Path) -> None:
    print(f"Best {result.agent} weights:")
    best_weights = result.train_config.get("best_weights", {}) if result.train_config else {}
    for name, value in best_weights.items():
        print(f"  {name}: {value:.6f}")
    print(
        "Best evaluation: "
        f"mean_score={result.best_evaluation.mean_score:.2f}, "
        f"max_tile={result.best_evaluation.max_tile}, "
        f"mean_steps={result.best_evaluation.mean_steps:.2f}"
    )
    print(f"Artifacts written to: {out_dir}")


def main() -> None:
    args = parse_args()
    config, policy_config, bounds = make_config(args)
    out_dir = Path(config.out_dir)
    spec = make_agent_spec(
        agent=config.agent,
        parameter_bounds=bounds,
        policy_config=policy_config,
    )
    train_config = {
        **asdict(config),
        "parameter_bounds": {name: list(value) for name, value in bounds.items()},
    }

    print(
        "Starting genetic optimization: "
        f"agent={config.agent}, "
        f"generations={config.generations}, "
        f"population_size={config.population_size}, "
        f"episodes_per_candidate={config.episodes_per_candidate}, "
        f"total_candidates={config.generations * config.population_size}, "
        f"workers={config.workers}, "
        f"seed={config.seed}",
        flush=True,
    )

    optimizer = GeneticOptimizer(config=config, spec=spec, train_config=train_config)
    result = optimizer.run(verbose=True, progress=not args.no_progress)
    best_weights = spec.weights_to_dict(result.best_weights)
    result = EvolutionResult(
        best_weights=result.best_weights,
        best_evaluation=result.best_evaluation,
        best_vector=result.best_vector,
        history=result.history,
        agent=result.agent,
        policy_config={
            **(result.policy_config or {}),
            "weights": best_weights,
        },
        train_config={
            **(result.train_config or {}),
            "best_weights": best_weights,
        },
    )

    save_best_parameters(result, config, out_dir / f"best_{config.agent}_weights.json")
    save_history(result, out_dir / "evolution_history.csv")
    plot_history(
        result.history,
        out_dir / "parameter_evolution.png",
        parameter_names=spec.parameter_names,
        parameter_bounds=spec.parameter_bounds,
    )
    print_result(result, out_dir)


if __name__ == "__main__":
    main()
