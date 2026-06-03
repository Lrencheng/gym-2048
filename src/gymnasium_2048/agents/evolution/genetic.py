from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import numpy as np
from tqdm import tqdm

from gymnasium_2048.agents.evolution.config import (
    PARAMETER_BOUNDS,
    PARAMETER_NAMES,
    EvolutionConfig,
    load_evolution_config,
)
from gymnasium_2048.agents.evolution.evaluation import (
    EvaluationResult,
    evaluate_weights,
    make_episode_seeds,
)
from gymnasium_2048.agents.evolution.specs import AgentSpec, make_agent_spec
from gymnasium_2048.agents.heuristic import HeuristicWeights

CandidateEvaluator = Callable[[HeuristicWeights, Sequence[int], str], EvaluationResult]
ProgressBar = Any | None


def _evaluate_candidate_worker(
    candidate_index: int,
    vector: np.ndarray,
    episode_seeds: Sequence[int],
    env_id: str,
    spec: AgentSpec,
) -> tuple[int, EvaluationResult]:
    weights = spec.vector_to_weights(vector)
    return (
        candidate_index,
        evaluate_weights(
            weights,
            episode_seeds,
            env_id,
            spec=spec,
        ),
    )


@dataclass(frozen=True)
class GenerationRecord:
    generation: int
    best_fitness: float
    mean_fitness: float
    best_mean_score: float
    best_max_tile: int
    best_mean_steps: float
    best_vector: np.ndarray


@dataclass(frozen=True)
class EvolutionResult:
    best_weights: HeuristicWeights
    best_evaluation: EvaluationResult
    best_vector: np.ndarray
    history: list[GenerationRecord]
    agent: str = "heuristic"
    policy_config: dict[str, Any] | None = None
    train_config: dict[str, Any] | None = None
    final_population: np.ndarray | None = None
    final_evaluations: list[EvaluationResult] | None = None


class GeneticOptimizer:
    def __init__(
        self,
        config: EvolutionConfig,
        evaluator: CandidateEvaluator | None = None,
        spec: AgentSpec | None = None,
        train_config: dict[str, Any] | None = None,
        initial_population: np.ndarray | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.spec = spec or self._make_default_spec(config.agent)
        self.evaluator = evaluator
        self.rng = np.random.default_rng(config.seed)
        self.train_config = dict(train_config or {})
        self.initial_population = (
            self._validate_initial_population(initial_population)
            if initial_population is not None
            else None
        )
        self.episode_seeds = make_episode_seeds(
            seed=config.seed + 1,
            episodes=config.episodes_per_candidate,
        )

    @staticmethod
    def _make_default_spec(agent: str) -> AgentSpec:
        if agent == "heuristic":
            return make_agent_spec(
                agent="heuristic",
                parameter_bounds=dict(zip(PARAMETER_NAMES, PARAMETER_BOUNDS.tolist())),
            )
        _config, policy_config, parameter_bounds = load_evolution_config(agent)
        return make_agent_spec(
            agent=agent,
            parameter_bounds=parameter_bounds,
            policy_config=policy_config,
        )

    def _validate_initial_population(self, population: np.ndarray) -> np.ndarray:
        initial = np.asarray(population, dtype=np.float64)
        expected_shape = (
            self.config.population_size,
            len(self.spec.parameter_names),
        )
        if initial.shape != expected_shape:
            raise ValueError(
                "initial_population must have shape "
                f"{expected_shape}, got {initial.shape}"
            )
        return np.array(
            [self.spec.clip_vector(vector) for vector in initial],
            dtype=np.float64,
        )

    def run(self, verbose: bool = False, progress: bool = False) -> EvolutionResult:
        population = (
            self.initial_population.copy()
            if self.initial_population is not None
            else self._make_initial_population()
        )
        history = []
        best_vector = population[0].copy()
        best_evaluation: EvaluationResult | None = None
        final_population: np.ndarray | None = None
        final_evaluations: list[EvaluationResult] | None = None
        progress_bar = self._make_progress_bar() if progress else None

        try:
            for generation in range(self.config.generations):
                evaluations = self._evaluate_population(
                    population=population,
                    generation=generation,
                    progress_bar=progress_bar,
                )
                final_population = population.copy()
                final_evaluations = list(evaluations)
                fitnesses = np.array(
                    [evaluation.fitness for evaluation in evaluations],
                    dtype=np.float64,
                )
                best_index = int(np.argmax(fitnesses))
                generation_best = evaluations[best_index]

                if (
                    best_evaluation is None
                    or generation_best.fitness > best_evaluation.fitness
                ):
                    best_evaluation = generation_best
                    best_vector = population[best_index].copy()

                record = GenerationRecord(
                    generation=generation,
                    best_fitness=float(generation_best.fitness),
                    mean_fitness=float(np.mean(fitnesses)),
                    best_mean_score=float(generation_best.mean_score),
                    best_max_tile=int(generation_best.max_tile),
                    best_mean_steps=float(generation_best.mean_steps),
                    best_vector=population[best_index].copy(),
                )
                history.append(record)
                self._update_generation_progress(progress_bar, record)

                if verbose:
                    self._print_generation(record, progress_bar)

                if generation + 1 < self.config.generations:
                    population = self._make_next_population(population, fitnesses)
        finally:
            if progress_bar is not None:
                progress_bar.close()

        assert best_evaluation is not None
        assert final_population is not None
        assert final_evaluations is not None
        return EvolutionResult(
            best_weights=self.spec.vector_to_weights(best_vector),
            best_evaluation=best_evaluation,
            best_vector=best_vector,
            history=history,
            agent=self.spec.agent,
            policy_config=dict(self.spec.policy_config),
            train_config=dict(self.train_config),
            final_population=final_population,
            final_evaluations=final_evaluations,
        )

    def _make_progress_bar(self) -> tqdm:
        total_candidates = self.config.generations * self.config.population_size
        return tqdm(
            total=total_candidates,
            desc="Evolution",
            unit="candidate",
        )

    def _make_initial_population(self) -> np.ndarray:
        lower = self.spec.parameter_bounds[:, 0]
        upper = self.spec.parameter_bounds[:, 1]
        return self.rng.uniform(
            low=lower,
            high=upper,
            size=(self.config.population_size, len(self.spec.parameter_bounds)),
        )

    def _evaluate_population(
        self,
        population: np.ndarray,
        generation: int,
        progress_bar: ProgressBar = None,
    ) -> list[EvaluationResult]:
        if self.config.workers > 1 and self.evaluator is None:
            return self._evaluate_population_parallel(
                population=population,
                generation=generation,
                progress_bar=progress_bar,
            )

        evaluations = []

        for candidate_index, vector in enumerate(population):
            self._update_candidate_progress(
                progress_bar=progress_bar,
                generation=generation,
                candidate_index=candidate_index,
            )
            weights = self.spec.vector_to_weights(vector)
            if self.evaluator is None:
                evaluation = evaluate_weights(
                    weights,
                    self.episode_seeds,
                    self.config.env_id,
                    spec=self.spec,
                )
            else:
                evaluation = self.evaluator(
                    weights,
                    self.episode_seeds,
                    self.config.env_id,
                )
            evaluations.append(evaluation)
            self._advance_candidate_progress(progress_bar, evaluation)

        return evaluations

    def _evaluate_population_parallel(
        self,
        population: np.ndarray,
        generation: int,
        progress_bar: ProgressBar = None,
    ) -> list[EvaluationResult]:
        worker_count = min(self.config.workers, len(population))
        evaluations: list[EvaluationResult | None] = [None] * len(population)
        self._update_parallel_progress(
            progress_bar=progress_bar,
            generation=generation,
            worker_count=worker_count,
        )

        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    _evaluate_candidate_worker,
                    candidate_index,
                    vector.copy(),
                    self.episode_seeds,
                    self.config.env_id,
                    self.spec,
                )
                for candidate_index, vector in enumerate(population)
            ]

            for future in as_completed(futures):
                candidate_index, evaluation = future.result()
                evaluations[candidate_index] = evaluation
                self._advance_candidate_progress(progress_bar, evaluation)

        return [evaluation for evaluation in evaluations if evaluation is not None]

    def _make_next_population(
        self,
        population: np.ndarray,
        fitnesses: np.ndarray,
    ) -> np.ndarray:
        elite_indices = np.argsort(fitnesses)[-self.config.elite_size :]
        next_population = [population[index].copy() for index in elite_indices]

        while len(next_population) < self.config.population_size:
            parent_a = self._select_parent(population, fitnesses)
            parent_b = self._select_parent(population, fitnesses)
            child = self._crossover(parent_a, parent_b)
            child = self._mutate(child)
            next_population.append(child)

        return np.array(next_population, dtype=np.float64)

    def _select_parent(
        self,
        population: np.ndarray,
        fitnesses: np.ndarray,
    ) -> np.ndarray:
        candidate_indices = self.rng.choice(
            len(population),
            size=self.config.tournament_size,
            replace=False,
        )
        best_candidate = candidate_indices[int(np.argmax(fitnesses[candidate_indices]))]
        return population[best_candidate]

    def _crossover(self, parent_a: np.ndarray, parent_b: np.ndarray) -> np.ndarray:
        if self.rng.random() > self.config.crossover_rate:
            return parent_a.copy()

        alpha = self.rng.random(len(parent_a))
        child = alpha * parent_a + (1.0 - alpha) * parent_b
        return self.spec.clip_vector(child)

    def _mutate(self, vector: np.ndarray) -> np.ndarray:
        span = self.spec.parameter_bounds[:, 1] - self.spec.parameter_bounds[:, 0]
        mutation_mask = self.rng.random(len(vector)) < self.config.mutation_rate
        noise = self.rng.normal(loc=0.0, scale=self.config.mutation_scale * span)
        mutated = vector + mutation_mask * noise
        return self.spec.clip_vector(mutated)

    def _update_candidate_progress(
        self,
        progress_bar: ProgressBar,
        generation: int,
        candidate_index: int,
    ) -> None:
        if progress_bar is None:
            return

        progress_bar.set_description(
            (
                f"Generation {generation + 1}/{self.config.generations} "
                f"candidate {candidate_index + 1}/{self.config.population_size}"
            )
        )

    def _update_parallel_progress(
        self,
        progress_bar: ProgressBar,
        generation: int,
        worker_count: int,
    ) -> None:
        if progress_bar is None:
            return

        progress_bar.set_description(
            (
                f"Generation {generation + 1}/{self.config.generations} "
                f"parallel workers {worker_count}"
            )
        )

    @staticmethod
    def _advance_candidate_progress(
        progress_bar: ProgressBar,
        evaluation: EvaluationResult,
    ) -> None:
        if progress_bar is None:
            return

        progress_bar.update(1)
        progress_bar.set_postfix(
            {
                "score": f"{evaluation.mean_score:.0f}",
                "tile": evaluation.max_tile,
            }
        )

    def _update_generation_progress(
        self,
        progress_bar: ProgressBar,
        record: GenerationRecord,
    ) -> None:
        if progress_bar is None:
            return

        progress_bar.set_postfix(
            {
                "gen": f"{record.generation + 1}/{self.config.generations}",
                "best": f"{record.best_mean_score:.0f}",
                "tile": record.best_max_tile,
            }
        )

    @staticmethod
    def _print_generation(
        record: GenerationRecord,
        progress_bar: ProgressBar = None,
    ) -> None:
        message = (
            f"generation={record.generation} "
            f"best_score={record.best_mean_score:.2f} "
            f"mean_fitness={record.mean_fitness:.2f} "
            f"max_tile={record.best_max_tile} "
            f"mean_steps={record.best_mean_steps:.2f}"
        )
        if progress_bar is not None:
            progress_bar.write(message)
        else:
            print(message)
