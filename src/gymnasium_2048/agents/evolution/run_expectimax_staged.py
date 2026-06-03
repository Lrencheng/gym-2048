from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from gymnasium_2048.agents.evolution.config import EvolutionConfig, load_evolution_config
from gymnasium_2048.agents.evolution.evaluation import (
    EvaluationResult,
    evaluate_weights,
    make_episode_seeds,
)
from gymnasium_2048.agents.evolution.genetic import (
    EvolutionResult,
    GeneticOptimizer,
    _evaluate_candidate_worker,
)
from gymnasium_2048.agents.evolution.plotting import plot_history
from gymnasium_2048.agents.evolution.run_evolution import (
    save_best_parameters,
    save_history,
)
from gymnasium_2048.agents.evolution.specs import AgentSpec, make_agent_spec


@dataclass(frozen=True)
class StageConfig:
    name: str
    depth: int
    chance_samples: int | None
    episodes: int
    population_size: int
    generations: int
    elite_size: int
    tournament_size: int
    mutation_scale: float
    seed_elite_count: int = 4
    seed_mutation_scale: float = 0.08


@dataclass(frozen=True)
class FinalConfig:
    depth: int
    chance_samples: int | None
    episodes: int
    top_k: int


DEFAULT_STAGES = (
    StageConfig(
        name="stage1",
        depth=1,
        chance_samples=2,
        episodes=5,
        population_size=12,
        generations=5,
        elite_size=3,
        tournament_size=3,
        mutation_scale=0.12,
    ),
    StageConfig(
        name="stage2",
        depth=2,
        chance_samples=4,
        episodes=15,
        population_size=12,
        generations=5,
        elite_size=3,
        tournament_size=3,
        mutation_scale=0.08,
        seed_elite_count=4,
        seed_mutation_scale=0.06,
    ),
)

DEFAULT_FINAL = FinalConfig(
    depth=2,
    chance_samples=6,
    episodes=50,
    top_k=5,
)


def select_top_vectors(
    population: np.ndarray,
    evaluations: Sequence[EvaluationResult],
    top_k: int,
) -> np.ndarray:
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if len(population) != len(evaluations):
        raise ValueError("population and evaluations must have the same length")

    fitnesses = np.array([evaluation.fitness for evaluation in evaluations])
    top_indices = np.argsort(fitnesses)[-top_k:][::-1]
    return np.asarray(population, dtype=np.float64)[top_indices].copy()


def seed_population_from_elites(
    elites: np.ndarray,
    population_size: int,
    mutation_scale: float,
    spec: AgentSpec,
    rng: np.random.Generator,
) -> np.ndarray:
    if population_size < 1:
        raise ValueError("population_size must be at least 1")
    elite_vectors = np.asarray(elites, dtype=np.float64)
    if elite_vectors.ndim != 2 or elite_vectors.shape[0] == 0:
        raise ValueError("elites must be a non-empty 2-D array")

    span = spec.parameter_bounds[:, 1] - spec.parameter_bounds[:, 0]
    population = []
    for vector in elite_vectors[:population_size]:
        population.append(spec.clip_vector(vector))

    elite_index = 0
    while len(population) < population_size:
        base = elite_vectors[elite_index % len(elite_vectors)]
        noise = rng.normal(loc=0.0, scale=mutation_scale * span)
        population.append(spec.clip_vector(base + noise))
        elite_index += 1

    return np.asarray(population, dtype=np.float64)


def evaluate_vectors(
    vectors: np.ndarray,
    spec: AgentSpec,
    config: EvolutionConfig,
    episode_seeds: Sequence[int],
    workers: int,
) -> list[EvaluationResult]:
    if workers <= 1:
        return [
            evaluate_weights(
                spec.vector_to_weights(vector),
                episode_seeds,
                config.env_id,
                spec=spec,
            )
            for vector in vectors
        ]

    from concurrent.futures import ProcessPoolExecutor, as_completed

    evaluations: list[EvaluationResult | None] = [None] * len(vectors)
    worker_count = min(workers, len(vectors))
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                _evaluate_candidate_worker,
                index,
                vector.copy(),
                episode_seeds,
                config.env_id,
                spec,
            )
            for index, vector in enumerate(vectors)
        ]
        for future in as_completed(futures):
            index, evaluation = future.result()
            evaluations[index] = evaluation

    return [evaluation for evaluation in evaluations if evaluation is not None]


def _result_with_best_weights(result: EvolutionResult, spec: AgentSpec) -> EvolutionResult:
    best_weights = spec.weights_to_dict(result.best_weights)
    return EvolutionResult(
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
        final_population=result.final_population,
        final_evaluations=result.final_evaluations,
    )


def save_stage_artifacts(
    stage_name: str,
    result: EvolutionResult,
    config: EvolutionConfig,
    spec: AgentSpec,
    out_dir: Path,
) -> None:
    stage_dir = out_dir / stage_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    printable_result = _result_with_best_weights(result, spec)
    save_best_parameters(
        printable_result,
        config,
        stage_dir / "best_expectimax_weights.json",
    )
    save_history(printable_result, stage_dir / "evolution_history.csv")
    plot_history(
        printable_result.history,
        stage_dir / "parameter_evolution.png",
        parameter_names=spec.parameter_names,
        parameter_bounds=spec.parameter_bounds,
    )


def _stage_policy_config(
    base_policy_config: dict[str, object],
    depth: int,
    chance_samples: int | None,
) -> dict[str, object]:
    return {
        **base_policy_config,
        "depth": depth,
        "chance_samples": chance_samples,
    }


def _stage_train_config(
    config: EvolutionConfig,
    stage: StageConfig,
    parameter_bounds: dict[str, tuple[float, float]],
) -> dict[str, object]:
    return {
        **asdict(config),
        "stage": asdict(stage),
        "parameter_bounds": {
            name: list(value) for name, value in parameter_bounds.items()
        },
    }


def _make_stage_config(
    base_config: EvolutionConfig,
    stage: StageConfig,
    seed: int,
    workers: int,
    out_dir: Path,
) -> EvolutionConfig:
    return EvolutionConfig(
        agent="expectimax",
        env_id=base_config.env_id,
        out_dir=str(out_dir / stage.name),
        seed=seed,
        population_size=stage.population_size,
        generations=stage.generations,
        episodes_per_candidate=stage.episodes,
        workers=workers,
        elite_size=stage.elite_size,
        tournament_size=stage.tournament_size,
        crossover_rate=base_config.crossover_rate,
        mutation_rate=base_config.mutation_rate,
        mutation_scale=stage.mutation_scale,
    )


def _save_final_ranking(
    output_path: Path,
    vectors: np.ndarray,
    evaluations: Sequence[EvaluationResult],
    spec: AgentSpec,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ranking = sorted(
        enumerate(evaluations),
        key=lambda item: item[1].fitness,
        reverse=True,
    )
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "rank",
                "fitness",
                "mean_score",
                "max_tile",
                "mean_steps",
                "weights",
            ]
        )
        for rank, (index, evaluation) in enumerate(ranking, start=1):
            weights = spec.weights_to_dict(spec.vector_to_weights(vectors[index]))
            writer.writerow(
                [
                    rank,
                    evaluation.fitness,
                    evaluation.mean_score,
                    evaluation.max_tile,
                    evaluation.mean_steps,
                    json.dumps(weights, sort_keys=True),
                ]
            )


def _save_summary(
    output_path: Path,
    stage_results: list[dict[str, object]],
    final_vectors: np.ndarray,
    final_evaluations: Sequence[EvaluationResult],
    spec: AgentSpec,
    final_config: FinalConfig,
) -> None:
    best_index = int(np.argmax([evaluation.fitness for evaluation in final_evaluations]))
    best_weights = spec.weights_to_dict(spec.vector_to_weights(final_vectors[best_index]))
    payload = {
        "agent": "expectimax",
        "stages": stage_results,
        "final": {
            "config": asdict(final_config),
            "best_index": best_index,
            "best_evaluation": asdict(final_evaluations[best_index]),
            "best_weights": best_weights,
            "ranking": [
                {
                    "index": int(index),
                    "evaluation": asdict(evaluation),
                    "weights": spec.weights_to_dict(
                        spec.vector_to_weights(final_vectors[index])
                    ),
                }
                for index, evaluation in sorted(
                    enumerate(final_evaluations),
                    key=lambda item: item[1].fitness,
                    reverse=True,
                )
            ],
        },
    }
    output_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def run_staged_expectimax(
    out_dir: str | Path = "models/evolution/expectimax_staged",
    seed: int | None = None,
    workers: int | None = None,
    progress: bool = True,
    stages: Sequence[StageConfig] = DEFAULT_STAGES,
    final_config: FinalConfig = DEFAULT_FINAL,
) -> dict[str, object]:
    out_path = Path(out_dir)
    base_config, base_policy_config, parameter_bounds = load_evolution_config(
        "expectimax"
    )
    run_seed = base_config.seed if seed is None else int(seed)
    worker_count = base_config.workers if workers is None else int(workers)
    rng = np.random.default_rng(run_seed)
    stage_results = []
    previous_population: np.ndarray | None = None
    previous_evaluations: list[EvaluationResult] | None = None

    for stage_index, stage in enumerate(stages):
        stage_seed = run_seed + stage_index * 1000
        policy_config = _stage_policy_config(
            base_policy_config,
            depth=stage.depth,
            chance_samples=stage.chance_samples,
        )
        spec = make_agent_spec(
            agent="expectimax",
            parameter_bounds=parameter_bounds,
            policy_config=policy_config,
        )
        stage_config = _make_stage_config(
            base_config=base_config,
            stage=stage,
            seed=stage_seed,
            workers=worker_count,
            out_dir=out_path,
        )
        initial_population = None
        if previous_population is not None and previous_evaluations is not None:
            elites = select_top_vectors(
                previous_population,
                previous_evaluations,
                top_k=min(stage.seed_elite_count, len(previous_population)),
            )
            initial_population = seed_population_from_elites(
                elites=elites,
                population_size=stage.population_size,
                mutation_scale=stage.seed_mutation_scale,
                spec=spec,
                rng=rng,
            )

        optimizer = GeneticOptimizer(
            config=stage_config,
            spec=spec,
            train_config=_stage_train_config(
                config=stage_config,
                stage=stage,
                parameter_bounds=parameter_bounds,
            ),
            initial_population=initial_population,
        )
        result = optimizer.run(verbose=True, progress=progress)
        save_stage_artifacts(stage.name, result, stage_config, spec, out_path)
        previous_population = result.final_population
        previous_evaluations = result.final_evaluations
        stage_results.append(
            {
                "name": stage.name,
                "config": asdict(stage_config),
                "policy_config": policy_config,
                "best_evaluation": asdict(result.best_evaluation),
                "best_weights": spec.weights_to_dict(result.best_weights),
                "artifacts_dir": str(out_path / stage.name),
            }
        )

    assert previous_population is not None
    assert previous_evaluations is not None
    final_vectors = select_top_vectors(
        previous_population,
        previous_evaluations,
        top_k=min(final_config.top_k, len(previous_population)),
    )
    final_policy_config = _stage_policy_config(
        base_policy_config,
        depth=final_config.depth,
        chance_samples=final_config.chance_samples,
    )
    final_spec = make_agent_spec(
        agent="expectimax",
        parameter_bounds=parameter_bounds,
        policy_config=final_policy_config,
    )
    final_eval_config = EvolutionConfig(
        agent="expectimax",
        env_id=base_config.env_id,
        out_dir=str(out_path / "final"),
        seed=run_seed + 9000,
        population_size=max(len(final_vectors), 2),
        generations=1,
        episodes_per_candidate=final_config.episodes,
        workers=worker_count,
        elite_size=1,
        tournament_size=1,
    )
    final_seeds = make_episode_seeds(
        seed=final_eval_config.seed + 1,
        episodes=final_config.episodes,
    )
    final_evaluations = evaluate_vectors(
        vectors=final_vectors,
        spec=final_spec,
        config=final_eval_config,
        episode_seeds=final_seeds,
        workers=worker_count,
    )
    final_dir = out_path / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    best_final_index = int(
        np.argmax([evaluation.fitness for evaluation in final_evaluations])
    )
    final_result = EvolutionResult(
        best_weights=final_spec.vector_to_weights(final_vectors[best_final_index]),
        best_evaluation=final_evaluations[best_final_index],
        best_vector=final_vectors[best_final_index],
        history=[],
        agent="expectimax",
        policy_config={
            **final_policy_config,
            "weights": final_spec.weights_to_dict(
                final_spec.vector_to_weights(final_vectors[best_final_index])
            ),
        },
        train_config={
            **asdict(final_eval_config),
            "stage": "final",
            "best_weights": final_spec.weights_to_dict(
                final_spec.vector_to_weights(final_vectors[best_final_index])
            ),
        },
        final_population=final_vectors,
        final_evaluations=list(final_evaluations),
    )
    save_best_parameters(
        final_result,
        final_eval_config,
        final_dir / "best_expectimax_weights.json",
    )
    _save_final_ranking(
        final_dir / "final_ranking.csv",
        final_vectors,
        final_evaluations,
        final_spec,
    )
    _save_summary(
        out_path / "staged_expectimax_summary.json",
        stage_results,
        final_vectors,
        final_evaluations,
        final_spec,
        final_config,
    )
    return {
        "out_dir": str(out_path),
        "stage_results": stage_results,
        "final_best_weights": final_spec.weights_to_dict(final_result.best_weights),
        "final_best_evaluation": asdict(final_result.best_evaluation),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run staged Expectimax weight optimization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--out-dir",
        default="models/evolution/expectimax_staged",
        help="directory for staged training artifacts",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="disable progress bars",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_staged_expectimax(
        out_dir=args.out_dir,
        seed=args.seed,
        workers=args.workers,
        progress=not args.no_progress,
    )
    print(
        "Staged Expectimax evolution complete: "
        f"out_dir={result['out_dir']}, "
        f"best_score={result['final_best_evaluation']['mean_score']:.2f}"
    )


if __name__ == "__main__":
    main()
