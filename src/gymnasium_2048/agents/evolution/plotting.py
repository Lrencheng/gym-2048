from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from gymnasium_2048.agents.evolution.config import PARAMETER_BOUNDS, PARAMETER_NAMES
from gymnasium_2048.agents.evolution.genetic import GenerationRecord


def plot_history(
    history: Sequence[GenerationRecord],
    output_path: str | Path,
    parameter_names: Sequence[str] = PARAMETER_NAMES,
    parameter_bounds: np.ndarray = PARAMETER_BOUNDS,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generations = [record.generation for record in history]
    best_fitness = [record.best_fitness for record in history]
    mean_fitness = [record.mean_fitness for record in history]
    max_tiles = [record.best_max_tile for record in history]
    mean_steps = [record.best_mean_steps for record in history]
    best_vectors = np.array([record.best_vector for record in history])

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    axs[0, 0].plot(generations, best_fitness, marker="o", label="best")
    axs[0, 0].plot(generations, mean_fitness, marker="s", label="mean")
    axs[0, 0].set_title("Fitness")
    axs[0, 0].set_xlabel("Generation")
    axs[0, 0].set_ylabel("Mean score")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    axs[0, 1].plot(generations, max_tiles, marker="o", color="tab:green")
    axs[0, 1].set_title("Best Max Tile")
    axs[0, 1].set_xlabel("Generation")
    axs[0, 1].set_ylabel("Tile value")
    axs[0, 1].grid(True)

    axs[1, 0].plot(generations, mean_steps, marker="o", color="tab:orange")
    axs[1, 0].set_title("Best Mean Steps")
    axs[1, 0].set_xlabel("Generation")
    axs[1, 0].set_ylabel("Steps")
    axs[1, 0].grid(True)

    bounds = np.asarray(parameter_bounds, dtype=np.float64)
    normalized_vectors = (best_vectors - bounds[:, 0]) / (
        bounds[:, 1] - bounds[:, 0]
    )
    for index, name in enumerate(parameter_names):
        axs[1, 1].plot(generations, normalized_vectors[:, index], label=name)
    axs[1, 1].set_title("Best Parameters")
    axs[1, 1].set_xlabel("Generation")
    axs[1, 1].set_ylabel("Normalized value")
    axs[1, 1].set_ylim(-0.05, 1.05)
    axs[1, 1].legend(fontsize=8)
    axs[1, 1].grid(True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
