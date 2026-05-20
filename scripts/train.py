import argparse
import logging
import os
from typing import Any

import gymnasium as gym
import numpy as np
from tqdm import trange

from gymnasium_2048.agents.ntuple import (
    NTupleNetworkBasePolicy,
    NTupleNetworkQLearningPolicy,
    NTupleNetworkTDPolicy,
    NTupleNetworkTDPolicySmall,
)
from gymnasium_2048.agents.supervised_cnn import (
    SupervisedTrainingConfig,
    train_supervised_cnn,
)

logging.basicConfig(
    filename="train.log",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train 2048 agents",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--algo",
        "--agent",
        dest="algo",
        default="tdl",
        help="agent or RL algorithm",
        choices=["ql", "tdl", "tdl-small", "supervised_cnn"],
    )
    parser.add_argument(
        "--env",
        default="gymnasium_2048:gymnasium_2048/TwentyFortyEight-v0",
        help="environment id",
    )
    parser.add_argument(
        "-i",
        "--trained-agent",
        default="",
        help="path to a pretrained n-tuple agent to continue training",
    )
    parser.add_argument(
        "-n",
        "--n-episodes",
        type=int,
        default=100_000,
        help="number of n-tuple training episodes",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=5_000,
        help="evaluate the n-tuple agent every n episodes",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=1000,
        help="number of episodes to use for n-tuple evaluation",
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=-1,
        help="save the n-tuple model every n steps (if negative, final only)",
    )
    parser.add_argument("--seed", type=int, default=42, help="random generator seed")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0025,
        help="learning rate",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        "--out",
        dest="out_dir",
        default="models",
        help="path to the directory containing output models",
    )

    parser.add_argument("--data", help="Expectimax .npz data for supervised CNN")
    parser.add_argument("--epochs", type=int, default=1, help="supervised epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="supervised batch size")
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1.0e-4,
        help="supervised AdamW weight decay",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="soft-label distillation temperature",
    )
    parser.add_argument("--device", default="cpu", help="supervised training device")
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
        help="fraction of data used for validation",
    )
    parser.add_argument(
        "--no-high-score-weighting",
        action="store_true",
        help="disable conservative high-score sample weighting",
    )
    parser.add_argument(
        "--score-weight",
        type=float,
        default=0.5,
        help="final-score component for supervised sample weighting",
    )
    parser.add_argument(
        "--tile-weight",
        type=float,
        default=0.4,
        help="final-max-tile component for supervised sample weighting",
    )
    parser.add_argument(
        "--late-game-weight",
        type=float,
        default=0.2,
        help="step-index component for supervised sample weighting",
    )
    parser.add_argument(
        "--difficulty-weight",
        type=float,
        default=0.2,
        help="low-empty-cell component for supervised sample weighting",
    )
    parser.add_argument(
        "--max-sample-weight",
        type=float,
        default=3.0,
        help="clip value before supervised sample weights are normalized",
    )
    return parser.parse_args()


def make_policy(algo: str, trained_agent: str) -> NTupleNetworkBasePolicy:
    algo_policy_map = {
        "ql": NTupleNetworkQLearningPolicy,
        "tdl": NTupleNetworkTDPolicy,
        "tdl-small": NTupleNetworkTDPolicySmall,
    }
    policy = algo_policy_map[algo]
    return policy.load(trained_agent) if trained_agent else policy()


def play_game(
    env: gym.Env,
    policy: NTupleNetworkBasePolicy,
    learn: bool = False,
    learning_rate: float = 0.0025,
) -> dict[str, Any]:
    _observation, info = env.reset()
    terminated = truncated = False

    while not terminated and not truncated:
        state = info["board"]
        action = policy.predict(state=state)
        _observation, reward, terminated, truncated, info = env.step(action)
        if learn:
            policy.learn(
                state=state,
                action=action,
                reward=float(reward),
                next_state=info["board"],
                learning_rate=learning_rate,
            )

    return info


def evaluate(
    env: gym.Env,
    policy: NTupleNetworkBasePolicy,
    eval_episodes: int,
) -> dict[str, Any]:
    winning_rate = 0
    total_score = 0
    max_tile = 0

    with trange(eval_episodes, desc="Evaluate", unit="episode", leave=False) as pbar:
        for _ in pbar:
            info = play_game(env=env, policy=policy)
            winning_rate += int(2 ** info["max"] >= 2048)
            total_score += info["total_score"]
            max_tile = max(max_tile, 2 ** info["max"])
            pbar.set_postfix({"max_tile": max_tile})

    return {
        "winning_rate": winning_rate / eval_episodes,
        "mean_score": total_score / eval_episodes,
        "max_tile": max_tile,
    }


def log_eval_metrics(episode: int, metrics: dict[str, Any]) -> None:
    logger.info(
        "episode %d: winning rate = %.2f, mean score = %.2f, max tile = %d",
        episode,
        metrics["winning_rate"],
        metrics["mean_score"],
        metrics["max_tile"],
    )


def save_best_policy(out_dir: str, policy: NTupleNetworkBasePolicy) -> None:
    best_model_path = os.path.join(out_dir, "best_n_tuple_network_policy.zip")
    logger.info("new best model saved to %s", best_model_path)
    policy.save(path=best_model_path)


def save_checkpoint(
    episode: int,
    out_dir: str,
    policy: NTupleNetworkBasePolicy,
) -> None:
    checkpoint_path = os.path.join(out_dir, f"checkpoint_episode_{episode}.zip")
    logger.info("checkpoint saved to %s", checkpoint_path)
    policy.save(path=checkpoint_path)


def train_supervised_from_args(args: argparse.Namespace) -> None:
    if not args.data:
        raise ValueError("supervised_cnn training requires --data")

    config = SupervisedTrainingConfig(
        data_path=args.data,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        validation_fraction=args.validation_fraction,
        device=args.device,
        seed=args.seed,
        use_high_score_weighting=not args.no_high_score_weighting,
        score_weight=args.score_weight,
        tile_weight=args.tile_weight,
        late_game_weight=args.late_game_weight,
        difficulty_weight=args.difficulty_weight,
        max_sample_weight=args.max_sample_weight,
    )
    result = train_supervised_cnn(config)
    print(
        "Supervised CNN training complete: "
        f"best_validation_loss={result['best_validation_loss']:.6f}, "
        f"best={result['best_checkpoint']}, "
        f"last={result['last_checkpoint']}"
    )


def train() -> None:
    args = parse_args()

    if args.algo == "supervised_cnn":
        train_supervised_from_args(args)
        return

    np.random.seed(args.seed)
    env = gym.make(args.env)
    policy = make_policy(algo=args.algo, trained_agent=args.trained_agent)
    best_mean_score = 0
    os.makedirs(args.out_dir, exist_ok=True)

    logger.info("start training n-tuple network")

    with trange(1, args.n_episodes + 1, desc="Train", unit="episode") as pbar:
        for e in pbar:
            play_game(
                env=env,
                policy=policy,
                learn=True,
                learning_rate=args.learning_rate,
            )

            should_evaluate = args.eval_freq > 0 and (
                e % args.eval_freq == 0 or e == args.n_episodes
            )
            if should_evaluate:
                metrics = evaluate(
                    env=env,
                    policy=policy,
                    eval_episodes=args.eval_episodes,
                )
                log_eval_metrics(episode=e, metrics=metrics)

                if metrics["mean_score"] > best_mean_score:
                    best_mean_score = metrics["mean_score"]
                    save_best_policy(out_dir=args.out_dir, policy=policy)

                metrics["best_mean_score"] = best_mean_score
                pbar.set_postfix(metrics)

            should_save = e == args.n_episodes or (
                args.save_freq > 0 and e % args.save_freq == 0
            )
            if should_save:
                save_checkpoint(episode=e, out_dir=args.out_dir, policy=policy)

    env.close()
    logger.info("end training n-tuple network")


if __name__ == "__main__":
    train()
