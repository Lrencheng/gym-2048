import argparse

from gymnasium_2048.agents.expectimax import (
    augment_afterstate_samples,
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
    parser.add_argument(
        "--reward-transform",
        choices=["raw", "log2p1", "none"],
        default="raw",
        help="utility transform applied to merge rewards inside expectimax",
    )
    parser.add_argument(
        "--symmetry-augmentation",
        action="store_true",
        help="store all eight D4 transforms of every afterstate sample",
    )
    parser.add_argument(
        "--debug-fields",
        action="store_true",
        help="also store root_boards and target_qs",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        help="maximum samples per numbered NPZ shard",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="disable progress bar",
    )
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
        reward_transform=args.reward_transform,
        debug_fields=args.debug_fields,
    )
    if args.symmetry_augmentation:
        dataset = augment_afterstate_samples(dataset)
    saved_paths = save_expectimax_dataset(
        dataset,
        args.out,
        shard_size=args.shard_size,
    )
    metadata = dataset["metadata"]
    print(
        "Generated Expectimax afterstate dataset: "
        f"samples={metadata['num_samples']}, "
        f"roots={metadata['num_roots']}, "
        f"episodes={metadata['episodes']}, "
        f"depth={metadata['depth']}, "
        f"chance_samples={metadata['chance_samples']}, "
        f"workers={metadata['workers']}, "
        f"files={len(saved_paths)}, "
        f"out={args.out}"
    )


if __name__ == "__main__":
    main()
