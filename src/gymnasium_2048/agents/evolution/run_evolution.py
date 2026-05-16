from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from gymnasium_2048.agents.evolution.config import DEFAULT_SEED, EvolutionConfig
from gymnasium_2048.agents.evolution.genetic import (
    EvolutionResult,
    GeneticOptimizer,
)
from gymnasium_2048.agents.evolution.parameters import weights_to_dict
from gymnasium_2048.agents.evolution.plotting import plot_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize 2048 heuristic weights with a genetic algorithm",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--env",
        default="gymnasium_2048:gymnasium_2048/TwentyFortyEight-v0",
        help="environment id",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--population-size", type=int, default=8)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--elite-size", type=int, default=2)
    parser.add_argument("--tournament-size", type=int, default=3)
    parser.add_argument("--crossover-rate", type=float, default=0.8)
    parser.add_argument("--mutation-rate", type=float, default=0.25)
    parser.add_argument("--mutation-scale", type=float, default=0.15)
    parser.add_argument(
        "--out-dir",
        default="models/evolution",
        help="directory for best parameters, history csv and plot",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="disable the terminal progress bar",
    )
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> EvolutionConfig:
    return EvolutionConfig(
        env_id=args.env,
        seed=args.seed,
        population_size=args.population_size,
        generations=args.generations,
        episodes_per_candidate=args.episodes,
        elite_size=args.elite_size,
        tournament_size=args.tournament_size,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        mutation_scale=args.mutation_scale,
    )


def save_best_parameters(
    result: EvolutionResult,
    config: EvolutionConfig,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": config.seed,
        "config": asdict(config),
        "fitness": result.best_evaluation.fitness,
        "mean_score": result.best_evaluation.mean_score,
        "max_tile": result.best_evaluation.max_tile,
        "mean_steps": result.best_evaluation.mean_steps,
        "weights": weights_to_dict(result.best_weights),
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
    print("Best heuristic weights:")
    for name, value in weights_to_dict(result.best_weights).items():
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
    config = make_config(args)
    out_dir = Path(args.out_dir)

    print(
        "Starting genetic optimization: "
        f"generations={config.generations}, "
        f"population_size={config.population_size}, "
        f"episodes_per_candidate={config.episodes_per_candidate}, "
        f"total_candidates={config.generations * config.population_size}, "
        f"seed={config.seed}",
        flush=True,
    )

    optimizer = GeneticOptimizer(config=config)
    result = optimizer.run(verbose=True, progress=not args.no_progress)

    save_best_parameters(result, config, out_dir / "best_heuristic_weights.json")
    save_history(result, out_dir / "evolution_history.csv")
    plot_history(result.history, out_dir / "parameter_evolution.png")
    print_result(result, out_dir)


if __name__ == "__main__":
    main()
