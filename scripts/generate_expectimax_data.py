import argparse

from gymnasium_2048.agents.expectimax import (
    generate_expectimax_dataset,
    save_expectimax_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate supervised data from an Expectimax teacher",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--env",
        default="gymnasium_2048:gymnasium_2048/TwentyFortyEight-v0",
        help="environment id",
    )
    parser.add_argument("--episodes", type=int, default=10, help="number of episodes")
    parser.add_argument("--depth", type=int, default=1, help="Expectimax search depth")
    parser.add_argument("--out", required=True, help="output .npz path")
    parser.add_argument("--seed", type=int, default=42, help="random generator seed")
    parser.add_argument(
        "--max-steps",
        type=int,
        help="optional per-episode step cap for smoke data generation",
    )
    parser.add_argument(
        "--chance-samples",
        type=int,
        help=(
            "sample this many empty cells at chance nodes when the empty-cell "
            "count is above --full-chance-empty-threshold; omit for exact enumeration"
        ),
    )
    parser.add_argument(
        "--full-chance-empty-threshold",
        type=int,
        default=6,
        help="empty-cell count at or below which chance nodes enumerate all cells",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="number of worker processes for parallel episode generation",
    )
    parser.add_argument("--no-progress", action="store_true", help="disable progress bar")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = generate_expectimax_dataset(
        env_id=args.env,
        episodes=args.episodes,
        depth=args.depth,
        seed=args.seed,
        max_steps=args.max_steps,
        progress=not args.no_progress,
        chance_samples=args.chance_samples,
        full_chance_empty_threshold=args.full_chance_empty_threshold,
        workers=args.workers,
    )
    save_expectimax_dataset(dataset, args.out)
    metadata = dataset["metadata"]
    print(
        "Generated Expectimax dataset: "
        f"samples={metadata['num_samples']}, "
        f"episodes={metadata['episodes']}, "
        f"depth={metadata['depth']}, "
        f"chance_samples={metadata['chance_samples']}, "
        f"workers={metadata['workers']}, "
        f"out={args.out}"
    )


if __name__ == "__main__":
    main()
