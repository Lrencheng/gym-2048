from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from gymnasium_2048.agents.expectimax.data import load_expectimax_dataset


def _statistics(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "maximum": float(np.max(array)),
    }


def summarize_expectimax_dataset(path: str | Path) -> dict[str, Any]:
    dataset = load_expectimax_dataset(path)
    episodes = np.asarray(dataset["episodes"], dtype=np.int64)
    root_ids = np.asarray(dataset["root_ids"], dtype=np.int64)
    return {
        "samples": int(len(np.asarray(dataset["target_us"]))),
        "roots": int(len(np.unique(root_ids))),
        "episodes": int(len(np.unique(episodes))),
        "depth": int(dataset["metadata"]["depth"]),
        "target_u": _statistics(np.asarray(dataset["target_us"])),
        "immediate_reward": _statistics(
            np.asarray(dataset["immediate_rewards"])
        ),
        "max_tile": _statistics(np.asarray(dataset["max_tiles"])),
        "empty_count": _statistics(np.asarray(dataset["empty_counts"])),
        "metadata": dataset["metadata"],
    }


def print_summary(summary: dict[str, Any]) -> None:
    print(
        "Expectimax afterstate dataset: "
        f"samples={summary['samples']}, "
        f"roots={summary['roots']}, "
        f"episodes={summary['episodes']}, "
        f"depth={summary['depth']}, "
        f"mean_target_u={summary['target_u']['mean']:.3f}, "
        f"mean_reward={summary['immediate_reward']['mean']:.3f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize an Expectimax afterstate-value dataset",
    )
    parser.add_argument("path", help="dataset NPZ file or shard directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print_summary(summarize_expectimax_dataset(args.path))


if __name__ == "__main__":
    main()
