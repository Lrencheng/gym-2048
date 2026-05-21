from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from gymnasium_2048.agents.expectimax.data import load_expectimax_dataset


def _percentiles(values: np.ndarray) -> dict[str, float]:
    return {
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
    }


def _distribution(values: np.ndarray) -> dict[int, int]:
    unique, counts = np.unique(values.astype(int), return_counts=True)
    return {int(value): int(count) for value, count in zip(unique, counts)}


def summarize_expectimax_dataset(path: str | Path) -> dict[str, Any]:
    dataset = load_expectimax_dataset(path)
    episode_ids = np.asarray(dataset["episodes"], dtype=np.int64)
    final_scores = np.asarray(dataset["final_scores"], dtype=np.float64)
    final_max_tiles = np.asarray(dataset["final_max_tiles"], dtype=np.int64)
    actions = np.asarray(dataset["actions"], dtype=np.int64)
    legal_masks = np.asarray(dataset["legal_masks"], dtype=bool)
    action_probs = np.asarray(dataset["action_probs"], dtype=np.float64)
    empty_counts = np.asarray(dataset["empty_counts"], dtype=np.float64)
    scores = np.asarray(dataset["scores"], dtype=np.float64)

    unique_episodes, first_indices, sample_counts = np.unique(
        episode_ids,
        return_index=True,
        return_counts=True,
    )
    episode_scores = final_scores[first_indices]
    episode_max_tiles = final_max_tiles[first_indices]

    legal_action_labels = legal_masks[np.arange(len(actions)), actions]
    legal_prob_mass = np.sum(np.where(legal_masks, action_probs, 0.0), axis=1)
    illegal_prob_mass = np.sum(np.where(legal_masks, 0.0, action_probs), axis=1)
    clipped_probs = np.clip(action_probs, 1.0e-12, 1.0)
    entropy = -np.sum(np.where(legal_masks, action_probs * np.log(clipped_probs), 0.0), axis=1)
    argmax_actions = np.argmax(np.where(legal_masks, action_probs, -np.inf), axis=1)

    summary = {
        "path": str(path),
        "metadata": dataset["metadata"],
        "episodes": int(len(unique_episodes)),
        "samples": int(len(actions)),
        "samples_per_episode": {
            "mean": float(np.mean(sample_counts)),
            "min": int(np.min(sample_counts)),
            "max": int(np.max(sample_counts)),
            **_percentiles(sample_counts.astype(float)),
        },
        "score": {
            "mean": float(np.mean(episode_scores)),
            "std": float(np.std(episode_scores)),
            "min": int(np.min(episode_scores)),
            "max": int(np.max(episode_scores)),
            **_percentiles(episode_scores),
        },
        "max_tile_distribution": _distribution(episode_max_tiles),
        "reach_rates": {
            "512": float(np.mean(episode_max_tiles >= 512)),
            "1024": float(np.mean(episode_max_tiles >= 1024)),
            "2048": float(np.mean(episode_max_tiles >= 2048)),
            "4096": float(np.mean(episode_max_tiles >= 4096)),
            "8192": float(np.mean(episode_max_tiles >= 8192)),
        },
        "labels": {
            "illegal_action_labels": int(np.count_nonzero(~legal_action_labels)),
            "empty_legal_masks": int(np.count_nonzero(~legal_masks.any(axis=1))),
            "hard_action_matches_prob_argmax": float(np.mean(actions == argmax_actions)),
            "mean_legal_probability_mass": float(np.mean(legal_prob_mass)),
            "max_illegal_probability_mass": float(np.max(illegal_prob_mass)),
            "mean_policy_entropy": float(np.mean(entropy)),
        },
        "states": {
            "mean_empty_cells": float(np.mean(empty_counts)),
            "mean_score_before_action": float(np.mean(scores)),
            "max_score_before_action": int(np.max(scores)),
        },
    }
    return summary


def print_summary(summary: dict[str, Any]) -> None:
    print(f"Dataset: {summary['path']}")
    print(
        "Episodes/Samples: "
        f"episodes={summary['episodes']}, "
        f"samples={summary['samples']}, "
        f"mean_steps={summary['samples_per_episode']['mean']:.2f}"
    )
    print(
        "Score: "
        f"mean={summary['score']['mean']:.2f}, "
        f"std={summary['score']['std']:.2f}, "
        f"p50={summary['score']['p50']:.0f}, "
        f"p90={summary['score']['p90']:.0f}, "
        f"max={summary['score']['max']}"
    )
    print(f"Max tile distribution: {summary['max_tile_distribution']}")
    print(
        "Reach rates: "
        f"512={summary['reach_rates']['512']:.3f}, "
        f"1024={summary['reach_rates']['1024']:.3f}, "
        f"2048={summary['reach_rates']['2048']:.3f}, "
        f"4096={summary['reach_rates']['4096']:.3f}, "
        f"8192={summary['reach_rates']['8192']:.3f}"
    )
    print(
        "Label checks: "
        f"illegal_actions={summary['labels']['illegal_action_labels']}, "
        f"empty_masks={summary['labels']['empty_legal_masks']}, "
        f"hard_argmax_match={summary['labels']['hard_action_matches_prob_argmax']:.3f}, "
        f"mean_legal_prob_mass={summary['labels']['mean_legal_probability_mass']:.6f}, "
        f"max_illegal_prob_mass={summary['labels']['max_illegal_probability_mass']:.6f}, "
        f"mean_entropy={summary['labels']['mean_policy_entropy']:.3f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize an Expectimax teacher dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", required=True, help="path to Expectimax .npz dataset")
    parser.add_argument("--json-out", help="optional path to write full JSON summary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize_expectimax_dataset(args.data)
    print_summary(summary)
    if args.json_out:
        output_path = Path(args.json_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
