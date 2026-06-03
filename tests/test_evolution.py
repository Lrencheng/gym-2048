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

    def test_agent_specs_load_from_yaml_and_construct_policies(self):
        heuristic_config, heuristic_policy_config, heuristic_bounds = (
            load_evolution_config("heuristic")
        )
        self.assertEqual(heuristic_config.out_dir, "models/evolution/heuristic")
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
        )

        config, _policy_config, _bounds = make_cli_config(args)

        self.assertEqual(config.out_dir, "models/evolution/custom")

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


if __name__ == "__main__":
    unittest.main()
