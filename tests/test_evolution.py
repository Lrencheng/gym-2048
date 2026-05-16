import unittest
from contextlib import redirect_stderr
from io import StringIO

import numpy as np

from gymnasium_2048.agents.evolution import (
    PARAMETER_BOUNDS,
    EvolutionConfig,
    EvaluationResult,
    GeneticOptimizer,
    clip_vector,
    vector_to_weights,
    weights_to_vector,
)


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


if __name__ == "__main__":
    unittest.main()
