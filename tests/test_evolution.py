import unittest
from contextlib import redirect_stderr
from io import StringIO
from types import SimpleNamespace

import numpy as np
import pytest

from gymnasium_2048.agents.config import ConfigError
from gymnasium_2048.agents.evolution import (
    PARAMETER_BOUNDS,
    EvolutionConfig,
    EvaluationResult,
    GeneticOptimizer,
    clip_vector,
    load_evolution_config,
    make_agent_spec,
    vector_to_weights,
    weights_to_vector,
)
from gymnasium_2048.agents.expectimax import ExpectimaxPolicy
from gymnasium_2048.agents.evolution.run_evolution import make_config as make_cli_config
from gymnasium_2048.agents.evolution.run_expectimax_staged import (
    FinalConfig,
    StageConfig,
    run_staged_expectimax,
    seed_population_from_elites,
    select_top_vectors,
)
from gymnasium_2048.agents.heuristic import HeuristicPolicy


def fake_evaluator(weights, episode_seeds, env_id):
    vector = weights_to_vector(weights)
    target = PARAMETER_BOUNDS[:, 0] + 0.75 * (
        PARAMETER_BOUNDS[:, 1] - PARAMETER_BOUNDS[:, 0]
    )
    fitness = -float(np.sum((vector - target) ** 2))
    return EvaluationResult(
        fitness=fitness,
        mean_score=fitness,
        max_tile=128,
        mean_steps=float(len(episode_seeds)),
    )


class EvolutionTest(unittest.TestCase):
    def test_weight_vector_roundtrip(self):
        vector = np.array([12.0, 1.0, 2.0, 4.0, 6.0, 0.02, 1.5])

        weights = vector_to_weights(vector)

        np.testing.assert_allclose(weights_to_vector(weights), vector)

    def test_clip_vector_keeps_parameters_in_bounds(self):
        vector = np.array([-1.0, 99.0, 0.0, 7.0, 1.0, 0.5, 10.0])

        clipped = clip_vector(vector)

        self.assertTrue(np.all(clipped >= PARAMETER_BOUNDS[:, 0]))
        self.assertTrue(np.all(clipped <= PARAMETER_BOUNDS[:, 1]))

    def test_genetic_optimizer_is_reproducible(self):
        config = EvolutionConfig(
            seed=7,
            population_size=6,
            generations=3,
            episodes_per_candidate=2,
            elite_size=1,
            tournament_size=2,
        )

        first = GeneticOptimizer(config=config, evaluator=fake_evaluator).run()
        second = GeneticOptimizer(config=config, evaluator=fake_evaluator).run()

        self.assertEqual(len(first.history), config.generations)
        np.testing.assert_allclose(first.best_vector, second.best_vector)
        self.assertEqual(first.best_evaluation.fitness, second.best_evaluation.fitness)

    def test_progress_bar_mode_runs(self):
        config = EvolutionConfig(
            seed=7,
            population_size=2,
            generations=1,
            episodes_per_candidate=1,
            elite_size=1,
            tournament_size=1,
        )

        with redirect_stderr(StringIO()):
            result = GeneticOptimizer(config=config, evaluator=fake_evaluator).run(
                progress=True
            )

        self.assertEqual(len(result.history), 1)

    def test_initial_population_is_used_and_returned(self):
        config = EvolutionConfig(
            seed=7,
            population_size=2,
            generations=1,
            episodes_per_candidate=1,
            elite_size=1,
            tournament_size=1,
        )
        initial_population = np.array(
            [
                [12.0, 1.0, 2.0, 4.0, 6.0, 0.02, 1.5],
                [10.0, 1.2, 2.5, 3.5, 5.0, 0.05, 1.0],
            ],
            dtype=np.float64,
        )

        result = GeneticOptimizer(
            config=config,
            evaluator=fake_evaluator,
            initial_population=initial_population,
        ).run()

        np.testing.assert_allclose(result.final_population, initial_population)
        self.assertEqual(len(result.final_evaluations), config.population_size)

    def test_initial_population_shape_is_validated(self):
        config = EvolutionConfig(
            population_size=2,
            generations=1,
            episodes_per_candidate=1,
            elite_size=1,
            tournament_size=1,
        )

        with pytest.raises(ValueError):
            GeneticOptimizer(
                config=config,
                evaluator=fake_evaluator,
                initial_population=np.zeros((1, 7), dtype=np.float64),
            )

    def test_agent_specs_load_from_yaml_and_construct_policies(self):
        heuristic_config, heuristic_policy_config, heuristic_bounds = (
            load_evolution_config("heuristic")
        )
        self.assertEqual(heuristic_config.out_dir, "models/evolution/heuristic")
        self.assertEqual(heuristic_config.workers, 1)
        heuristic_spec = make_agent_spec(
            heuristic_config.agent,
            heuristic_bounds,
            heuristic_policy_config,
        )
        heuristic_weights = heuristic_spec.vector_to_weights(
            heuristic_spec.parameter_bounds[:, 0]
        )

        expectimax_config, expectimax_policy_config, expectimax_bounds = (
            load_evolution_config("expectimax")
        )
        self.assertEqual(expectimax_config.out_dir, "models/evolution/expectimax")
        self.assertEqual(expectimax_config.workers, 4)
        expectimax_spec = make_agent_spec(
            expectimax_config.agent,
            expectimax_bounds,
            expectimax_policy_config,
        )
        expectimax_weights = expectimax_spec.vector_to_weights(
            expectimax_spec.parameter_bounds[:, 0]
        )

        self.assertEqual(
            heuristic_spec.parameter_names,
            (
                "empty_cells",
                "smoothness",
                "monotonicity",
                "corner_max",
                "merge_potential",
                "edge_bonus",
                "reward",
            ),
        )
        self.assertEqual(
            expectimax_spec.parameter_names,
            (
                "empty_cells",
                "smoothness",
                "monotonicity",
                "merge_potential",
                "corner_max",
                "max_tile",
                "snake",
                "reward",
            ),
        )
        self.assertIsInstance(heuristic_spec.make_policy(heuristic_weights), HeuristicPolicy)
        self.assertIsInstance(
            expectimax_spec.make_policy(expectimax_weights),
            ExpectimaxPolicy,
        )

    def test_parameter_bounds_must_match_weight_fields(self):
        _config, policy_config, bounds = load_evolution_config("heuristic")
        invalid_bounds = dict(bounds)
        invalid_bounds.pop("reward")
        invalid_bounds["snake"] = (0.0, 1.0)

        with pytest.raises(ConfigError):
            make_agent_spec("heuristic", invalid_bounds, policy_config)

    def test_evolution_config_rejects_empty_output_dir(self):
        config = EvolutionConfig(out_dir="")

        with pytest.raises(ValueError):
            config.validate()

    def test_evolution_config_rejects_invalid_workers(self):
        config = EvolutionConfig(workers=0)

        with pytest.raises(ValueError):
            config.validate()

    def test_cli_out_dir_overrides_train_yaml(self):
        args = SimpleNamespace(
            agent="heuristic",
            config=None,
            env=None,
            seed=None,
            population_size=None,
            generations=None,
            episodes=None,
            elite_size=None,
            tournament_size=None,
            crossover_rate=None,
            mutation_rate=None,
            mutation_scale=None,
            out_dir="models/evolution/custom",
            workers=3,
        )

        config, _policy_config, _bounds = make_cli_config(args)

        self.assertEqual(config.out_dir, "models/evolution/custom")
        self.assertEqual(config.workers, 3)

    def test_small_real_optimization_runs_for_each_agent(self):
        for agent in ("heuristic", "expectimax"):
            _config, policy_config, bounds = load_evolution_config(agent)
            policy_config = dict(policy_config)
            policy_config["depth"] = 1
            config = EvolutionConfig(
                agent=agent,
                population_size=2,
                generations=1,
                episodes_per_candidate=1,
                elite_size=1,
                tournament_size=1,
                seed=11,
            )
            spec = make_agent_spec(agent, bounds, policy_config)

            result = GeneticOptimizer(config=config, spec=spec).run()

            self.assertEqual(len(result.history), 1)
            self.assertEqual(result.agent, agent)

    def test_parallel_real_optimization_runs(self):
        _config, policy_config, bounds = load_evolution_config("heuristic")
        config = EvolutionConfig(
            agent="heuristic",
            population_size=2,
            generations=1,
            episodes_per_candidate=1,
            elite_size=1,
            tournament_size=1,
            workers=2,
            seed=13,
        )
        spec = make_agent_spec("heuristic", bounds, policy_config)

        result = GeneticOptimizer(config=config, spec=spec).run()

        self.assertEqual(len(result.history), 1)
        self.assertEqual(result.agent, "heuristic")

    def test_select_top_vectors_orders_by_fitness(self):
        population = np.array(
            [
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
            ],
            dtype=np.float64,
        )
        evaluations = [
            EvaluationResult(fitness=1.0, mean_score=1.0, max_tile=2, mean_steps=1.0),
            EvaluationResult(fitness=3.0, mean_score=3.0, max_tile=4, mean_steps=1.0),
            EvaluationResult(fitness=2.0, mean_score=2.0, max_tile=8, mean_steps=1.0),
        ]

        selected = select_top_vectors(population, evaluations, top_k=2)

        np.testing.assert_allclose(selected, population[[1, 2]])

    def test_seed_population_from_elites_keeps_elites_and_clips_mutations(self):
        _config, policy_config, bounds = load_evolution_config("heuristic")
        spec = make_agent_spec("heuristic", bounds, policy_config)
        elites = np.array(
            [
                spec.parameter_bounds[:, 0],
                spec.parameter_bounds[:, 1],
            ],
            dtype=np.float64,
        )

        population = seed_population_from_elites(
            elites=elites,
            population_size=4,
            mutation_scale=0.5,
            spec=spec,
            rng=np.random.default_rng(5),
        )

        np.testing.assert_allclose(population[:2], elites)
        self.assertTrue(np.all(population >= spec.parameter_bounds[:, 0]))
        self.assertTrue(np.all(population <= spec.parameter_bounds[:, 1]))


def test_staged_expectimax_smoke_generates_artifacts(tmp_path):
    result = run_staged_expectimax(
        out_dir=tmp_path / "staged",
        seed=17,
        workers=1,
        progress=False,
        stages=(
            StageConfig(
                name="stage1",
                depth=1,
                chance_samples=1,
                episodes=1,
                population_size=2,
                generations=1,
                elite_size=1,
                tournament_size=1,
                mutation_scale=0.05,
                seed_elite_count=1,
            ),
            StageConfig(
                name="stage2",
                depth=1,
                chance_samples=1,
                episodes=1,
                population_size=2,
                generations=1,
                elite_size=1,
                tournament_size=1,
                mutation_scale=0.05,
                seed_elite_count=1,
            ),
        ),
        final_config=FinalConfig(
            depth=1,
            chance_samples=1,
            episodes=1,
            top_k=1,
        ),
    )

    output_dir = tmp_path / "staged"
    assert result["out_dir"] == str(output_dir)
    assert (output_dir / "stage1" / "parameter_evolution.png").exists()
    assert (output_dir / "stage2" / "parameter_evolution.png").exists()
    assert (output_dir / "final" / "final_ranking.csv").exists()
    assert (output_dir / "final" / "best_expectimax_weights.json").exists()
    assert (output_dir / "staged_expectimax_summary.json").exists()


if __name__ == "__main__":
    unittest.main()
