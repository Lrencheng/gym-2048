from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from gymnasium_2048.agents.config import DEFAULT_ENV_ID
from gymnasium_2048.agents.evaluation import EvaluationConfig, evaluate_config
from gymnasium_2048.agents.expectimax.board import max_tile_value
from gymnasium_2048.agents.expectimax.symmetry import apply_symmetry
from gymnasium_2048.agents.RLSL.replay import (
    REPLAY_SOURCE_ONLINE,
    REPLAY_SOURCE_TEACHER,
    RLSLReplayBuffer,
    ReplayAfterstateDataset,
    cap_current_samples,
    choose_admitted_samples,
)
from gymnasium_2048.agents.RLSL.search import choose_search_improved_action
from gymnasium_2048.agents.supervised_cnn import (
    CNNAfterstateEvaluator,
    CNNConfig,
    SupervisedCNN,
    encode_board,
    regression_loss,
    resolve_device,
)
from gymnasium_2048.agents.supervised_cnn.model import config_to_dict


@dataclass(frozen=True)
class EpisodeSample:
    afterstate: np.ndarray
    target_value: float


@dataclass(frozen=True)
class EpisodeRollout:
    samples: list[EpisodeSample]
    score: int
    max_tile: int
    length: int


@dataclass(frozen=True)
class TargetNormalization:
    mean: float
    std: float


@dataclass(frozen=True)
class RLSLTrainingConfig:
    out_dir: str
    env_id: str = DEFAULT_ENV_ID
    rounds: int = 20
    search_depth: int = 1
    num_eps_per_search: int = 1
    train_epochs_per_episode: int = 1
    batch_size: int = 8900
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-4
    l2_coeff: float | None = None
    loss_type: str = "mse"
    value_scale: float = 1.0
    target_mean: float | None = None
    target_std: float | None = None
    checkpoint_interval: int = 1
    eval_interval: int = 10
    eval_episodes: int = 10
    eval_workers: int = 1
    device: str = "auto"
    seed: int = 42
    num_workers: int = 0
    pin_memory: bool = True
    input_channels: int = 16
    conv_channels: int = 64
    hidden_size: int = 128
    chance_samples: int | None = None
    full_chance_empty_threshold: int = 6
    symmetry_augmentation: bool = True
    initial_checkpoint: str | None = None
    replay_enabled: bool = True
    replay_teacher_data_path: str | None = "data/expectimax_afterstates_400.npz"
    replay_capacity: int = 280_000
    replay_current_train_max_samples: int = 70_000
    replay_current_admission_fraction: float = 0.10
    replay_seed: int | None = None
    progress: bool = True


class _EpisodeAfterstateDataset(Dataset):
    def __init__(
        self,
        samples: list[EpisodeSample],
        *,
        num_channels: int,
        augment: bool,
        seed: int,
    ) -> None:
        self.samples = samples
        self.num_channels = int(num_channels)
        self.augment = bool(augment)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        board = np.asarray(sample.afterstate, dtype=np.uint8)
        if self.augment:
            board = apply_symmetry(board, int(self.rng.integers(8)))
        encoded = encode_board(board, num_channels=self.num_channels)
        return (
            torch.from_numpy(encoded),
            torch.tensor(float(sample.target_value), dtype=torch.float32),
        )


def _validate_config(config: RLSLTrainingConfig) -> None:
    if config.rounds < 1:
        raise ValueError("rounds must be at least 1")
    if config.search_depth != 1:
        raise ValueError("RLSL currently supports only search_depth=1")
    if config.num_eps_per_search < 1:
        raise ValueError("num_eps_per_search must be at least 1")
    if config.train_epochs_per_episode < 1:
        raise ValueError("train_epochs_per_episode must be at least 1")
    if config.batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if config.value_scale <= 0.0:
        raise ValueError("value_scale must be positive")
    if config.target_std is not None and config.target_std <= 0.0:
        raise ValueError("target_std must be positive when provided")
    if config.checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be at least 1")
    if config.eval_interval < 0:
        raise ValueError("eval_interval must be non-negative")
    if config.loss_type not in {"mse", "huber"}:
        raise ValueError("loss_type must be 'mse' or 'huber'")
    if config.full_chance_empty_threshold < 0:
        raise ValueError("full_chance_empty_threshold must be non-negative")
    if config.chance_samples is not None and config.chance_samples < 1:
        raise ValueError("chance_samples must be positive when provided")
    if config.replay_capacity < 1:
        raise ValueError("replay_capacity must be positive")
    if config.replay_current_train_max_samples < 0:
        raise ValueError("replay_current_train_max_samples must be non-negative")
    if not 0.0 <= config.replay_current_admission_fraction <= 1.0:
        raise ValueError("replay_current_admission_fraction must be in [0, 1]")
    if config.replay_enabled and not config.replay_teacher_data_path:
        raise ValueError("replay_teacher_data_path is required when replay is enabled")


def _effective_l2_coeff(config: RLSLTrainingConfig) -> float:
    return float(config.l2_coeff if config.l2_coeff is not None else 0.0)


def _resolve_target_normalization(
    config: RLSLTrainingConfig,
    checkpoint_payload: dict[str, Any] | None = None,
) -> TargetNormalization:
    if config.target_mean is not None:
        mean = float(config.target_mean)
    elif checkpoint_payload is not None and "target_mean" in checkpoint_payload:
        mean = float(checkpoint_payload["target_mean"])
    else:
        mean = 0.0

    if config.target_std is not None:
        std = float(config.target_std)
    elif checkpoint_payload is not None and "target_std" in checkpoint_payload:
        std = float(checkpoint_payload["target_std"])
    else:
        std = float(config.value_scale)

    if std <= 0.0:
        raise ValueError("target normalization std must be positive")
    return TargetNormalization(mean=mean, std=std)


def _l2_penalty(model: torch.nn.Module) -> torch.Tensor:
    penalties = [
        parameter.float().pow(2).sum()
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    if not penalties:
        return torch.tensor(0.0)
    return torch.stack(penalties).sum()


def _make_old_evaluator(
    model: SupervisedCNN,
    *,
    device: torch.device,
    normalization: TargetNormalization,
) -> CNNAfterstateEvaluator:
    old_model = SupervisedCNN(model.config)
    state_dict = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    old_model.load_state_dict(state_dict)
    for parameter in old_model.parameters():
        parameter.requires_grad_(False)
    return CNNAfterstateEvaluator(
        old_model,
        target_mean=normalization.mean,
        target_std=normalization.std,
        device=device,
    )


def collect_search_improved_episode(
    config: RLSLTrainingConfig,
    evaluator: CNNAfterstateEvaluator,
    *,
    episode_seed: int,
    policy_rng: np.random.Generator,
) -> EpisodeRollout:
    env = gym.make(config.env_id)
    try:
        _observation, info = env.reset(seed=int(episode_seed))
        terminated = truncated = False
        step = 0
        samples: list[EpisodeSample] = []

        while not terminated and not truncated:
            decision = choose_search_improved_action(
                np.asarray(info["board"], dtype=np.uint8),
                evaluator=evaluator,
                search_depth=config.search_depth,
                rng=policy_rng,
                chance_samples=config.chance_samples,
                full_chance_empty_threshold=config.full_chance_empty_threshold,
            )
            if not np.any(decision.legal_mask):
                break
            samples.extend(
                EpisodeSample(
                    afterstate=action_sample.afterstate.copy(),
                    target_value=action_sample.target_value,
                )
                for action_sample in decision.action_samples
            )
            _observation, _reward, terminated, truncated, info = env.step(
                decision.action
            )
            step += 1

        return EpisodeRollout(
            samples=samples,
            score=int(info.get("total_score", 0)),
            max_tile=max_tile_value(np.asarray(info["board"], dtype=np.uint8)),
            length=step,
        )
    finally:
        env.close()


def train_on_afterstate_dataset(
    *,
    model: SupervisedCNN,
    dataset: Dataset,
    config: RLSLTrainingConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    normalization: TargetNormalization | None = None,
) -> dict[str, float | int]:
    if len(dataset) == 0:
        return {
            "samples": 0,
            "epochs": config.train_epochs_per_episode,
            "regression_loss": 0.0,
            "l2_loss": 0.0,
            "total_loss": 0.0,
            "train_loss": 0.0,
        }

    model.train()
    owned_optimizer = optimizer is None
    if optimizer is None:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    pin_memory = bool(config.pin_memory and device.type == "cuda")
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )
    iterator = range(1, config.train_epochs_per_episode + 1)
    if config.progress:
        iterator = tqdm(iterator, desc="  SL train", unit="epoch", leave=False)

    coeff = _effective_l2_coeff(config)
    target_normalization = normalization or _resolve_target_normalization(config)
    totals = {"regression_loss": 0.0, "l2_loss": 0.0, "total_loss": 0.0}
    total_observations = 0
    for _epoch in iterator:
        epoch_reg = 0.0
        epoch_l2 = 0.0
        epoch_total = 0.0
        epoch_samples = 0
        for boards, targets in loader:
            boards = boards.to(device)
            targets = (
                targets.to(device) - float(target_normalization.mean)
            ) / float(target_normalization.std)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(boards)
            regression = regression_loss(
                predictions,
                targets,
                kind=config.loss_type,
            )
            l2 = (
                _l2_penalty(model).to(device) * coeff
                if coeff != 0.0
                else torch.zeros((), device=device)
            )
            total = regression + l2
            total.backward()
            optimizer.step()

            batch_size = int(len(targets))
            totals["regression_loss"] += float(regression.detach().cpu()) * batch_size
            totals["l2_loss"] += float(l2.detach().cpu()) * batch_size
            totals["total_loss"] += float(total.detach().cpu()) * batch_size
            total_observations += batch_size
            epoch_reg += float(regression.detach().cpu()) * batch_size
            epoch_l2 += float(l2.detach().cpu()) * batch_size
            epoch_total += float(total.detach().cpu()) * batch_size
            epoch_samples += batch_size

        if config.progress:
            denom = max(epoch_samples, 1)
            iterator.set_postfix(
                {
                    "reg": f"{epoch_reg / denom:.4f}",
                    "l2": f"{epoch_l2 / denom:.4f}",
                    "total": f"{epoch_total / denom:.4f}",
                }
            )

    if owned_optimizer:
        del optimizer

    denominator = max(total_observations, 1)
    stats = {
        key: value / denominator
        for key, value in totals.items()
    }
    stats["train_loss"] = stats["total_loss"]
    stats["samples"] = len(dataset)
    stats["epochs"] = config.train_epochs_per_episode
    return stats


def train_on_episode_samples(
    *,
    model: SupervisedCNN,
    samples: list[EpisodeSample],
    config: RLSLTrainingConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    normalization: TargetNormalization | None = None,
) -> dict[str, float | int]:
    dataset = _EpisodeAfterstateDataset(
        samples,
        num_channels=config.input_channels,
        augment=config.symmetry_augmentation,
        seed=config.seed,
    )
    return train_on_afterstate_dataset(
        model=model,
        dataset=dataset,
        config=config,
        device=device,
        optimizer=optimizer,
        normalization=normalization,
    )


def _torch_load(path: str | Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _make_model(
    config: RLSLTrainingConfig,
    device: torch.device,
) -> tuple[SupervisedCNN, TargetNormalization]:
    payload: dict[str, Any] | None = None
    if config.initial_checkpoint is not None:
        payload = _torch_load(config.initial_checkpoint, device)
        model_config = CNNConfig(**payload.get("model_config", {}))
        model = SupervisedCNN(model_config)
        model.load_state_dict(payload["model_state_dict"])
    else:
        model = SupervisedCNN(
            CNNConfig(
                input_channels=config.input_channels,
                conv_channels=config.conv_channels,
                hidden_size=config.hidden_size,
            )
        )
    normalization = _resolve_target_normalization(config, payload)
    return model.to(device), normalization


def _make_replay_buffer(config: RLSLTrainingConfig) -> RLSLReplayBuffer | None:
    if not config.replay_enabled:
        return None
    replay_seed = config.seed if config.replay_seed is None else config.replay_seed
    return RLSLReplayBuffer.from_teacher_dataset(
        config.replay_teacher_data_path or "",
        capacity=config.replay_capacity,
        seed=int(replay_seed),
    )


def _save_checkpoint(
    path: Path,
    *,
    model: SupervisedCNN,
    optimizer: torch.optim.Optimizer,
    config: RLSLTrainingConfig,
    normalization: TargetNormalization,
    round_index: int,
    stats: dict[str, Any],
) -> None:
    torch.save(
        {
            "model_config": config_to_dict(model.config),
            "training_config": asdict(config),
            "target_mean": normalization.mean,
            "target_std": normalization.std,
            "round": int(round_index),
            "stats": stats,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def _write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _evaluate_round(
    *,
    config: RLSLTrainingConfig,
    checkpoint_path: Path,
    round_index: int,
    device: torch.device,
) -> dict[str, object]:
    eval_dir = Path(config.out_dir) / f"evaluate_{round_index}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    summary = evaluate_config(
        EvaluationConfig(
            agent="supervised_cnn",
            env_id=config.env_id,
            checkpoint=str(checkpoint_path),
            episodes=config.eval_episodes,
            seed=config.seed + round_index * 10_007,
            title=f"RLSL depth-0 evaluation round {round_index}",
            output_path=str(eval_dir / "evaluation.png"),
            plot=False,
            depth=0,
            device=str(device),
            workers=config.eval_workers,
        )
    )
    _write_json(eval_dir / "summary.json", summary)
    return summary


def train_rlsl(config: RLSLTrainingConfig) -> dict[str, Any]:
    _validate_config(config)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    rng = np.random.default_rng(config.seed)
    device = resolve_device(config.device)

    out_dir = Path(config.out_dir)
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "config.json", asdict(config))

    model, normalization = _make_model(config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    replay_buffer = _make_replay_buffer(config)
    replay_seed = config.seed if config.replay_seed is None else config.replay_seed
    replay_rng = np.random.default_rng(int(replay_seed))

    history: list[dict[str, Any]] = []
    best_metric = -float("inf")
    best_round: int | None = None
    best_checkpoint = checkpoint_dir / "best.pt"
    last_checkpoint = checkpoint_dir / "last.pt"

    for round_index in range(1, config.rounds + 1):
        # ── round header ──
        if config.progress:
            print(
                f"\n{'='*52}\n"
                f"  Round {round_index}/{config.rounds}\n"
                f"{'='*52}"
            )

        old_evaluator = _make_old_evaluator(
            model,
            device=device,
            normalization=normalization,
        )

        # ── search phase: run num_eps_per_search episodes ──
        rollouts: list[EpisodeRollout] = []
        all_samples: list[EpisodeSample] = []
        episode_scores: list[int] = []
        episode_max_tiles: list[int] = []
        episode_lengths: list[int] = []

        for ep_idx in range(1, config.num_eps_per_search + 1):
            episode_seed = int(rng.integers(0, 2**31 - 1))
            policy_rng = np.random.default_rng(
                config.seed + round_index * 1_000_003 + ep_idx * 9973
            )
            rollout = collect_search_improved_episode(
                config,
                old_evaluator,
                episode_seed=episode_seed,
                policy_rng=policy_rng,
            )
            rollouts.append(rollout)
            all_samples.extend(rollout.samples)
            episode_scores.append(rollout.score)
            episode_max_tiles.append(rollout.max_tile)
            episode_lengths.append(rollout.length)

        # Use the first episode's rollout for backward-compatible row fields
        first_rollout = rollouts[0]
        target_values = np.asarray(
            [sample.target_value for sample in all_samples],
            dtype=np.float64,
        )
        if config.progress:
            if len(all_samples) > 0:
                target_str = (
                    f"mean {np.mean(target_values):.2f}  "
                    f"std {np.std(target_values):.2f}  "
                    f"[{np.min(target_values):.2f}, {np.max(target_values):.2f}]"
                )
            else:
                target_str = "no samples"

            if config.num_eps_per_search > 1:
                print(
                    f"  Search complete ({config.num_eps_per_search} episodes):\n"
                    f"    scores      = {episode_scores}\n"
                    f"    max_tiles   = {episode_max_tiles}\n"
                    f"    steps       = {episode_lengths}\n"
                    f"    total steps = {sum(episode_lengths)}\n"
                    f"    dataset     = {len(all_samples)} samples\n"
                    f"    target_u    = {target_str}"
                )
            else:
                print(
                    "  Search complete:\n"
                    f"    score       = {first_rollout.score}\n"
                    f"    max_tile    = {first_rollout.max_tile}\n"
                    f"    steps       = {first_rollout.length}\n"
                    f"    dataset     = {len(all_samples)} samples\n"
                    f"    target_u    = {target_str}"
                )

        # ── supervised learning phase ──
        if replay_buffer is not None:
            current_train_samples = cap_current_samples(
                all_samples,
                max_samples=config.replay_current_train_max_samples,
                rng=replay_rng,
            )
            replay_train_dataset = ReplayAfterstateDataset(
                buffer=replay_buffer,
                current_samples=current_train_samples,
                config=config,
                seed=config.seed + round_index,
            )
            train_dataset_size = len(replay_train_dataset)
            train_stats = train_on_afterstate_dataset(
                model=model,
                dataset=replay_train_dataset,
                config=config,
                device=device,
                optimizer=optimizer,
                normalization=normalization,
            )
            admitted_samples = choose_admitted_samples(
                current_train_samples,
                fraction=config.replay_current_admission_fraction,
                rng=replay_rng,
            )
            replay_buffer.append_online(admitted_samples)
            replay_source_counts = replay_buffer.source_counts()
        else:
            current_train_samples = all_samples
            train_dataset_size = len(current_train_samples)
            train_stats = train_on_episode_samples(
                model=model,
                samples=current_train_samples,
                config=config,
                device=device,
                optimizer=optimizer,
                normalization=normalization,
            )
            admitted_samples = []
            replay_source_counts = {
                REPLAY_SOURCE_TEACHER: 0,
                REPLAY_SOURCE_ONLINE: 0,
            }

        if config.progress:
            print(
                "  SL complete:\n"
                f"    reg_loss    = {train_stats['regression_loss']:.6f}\n"
                f"    l2_loss     = {train_stats['l2_loss']:.6f}\n"
                f"    total_loss  = {train_stats['total_loss']:.6f}"
            )

        row: dict[str, Any] = {
            "round": round_index,
            "num_eps_per_search": config.num_eps_per_search,
            "episode_scores": episode_scores,
            "episode_max_tiles": episode_max_tiles,
            "episode_lengths": episode_lengths,
            "episode_score": first_rollout.score,
            "episode_score_mean": float(np.mean(episode_scores)) if episode_scores else 0,
            "max_tile": first_rollout.max_tile,
            "max_tile_all": max(episode_max_tiles) if episode_max_tiles else 0,
            "episode_length": first_rollout.length,
            "episode_length_total": sum(episode_lengths),
            "dataset_size": train_dataset_size,
            "episode_samples": len(all_samples),
            "replay_size": len(replay_buffer) if replay_buffer is not None else 0,
            "replay_teacher_samples": replay_source_counts[REPLAY_SOURCE_TEACHER],
            "replay_online_samples": replay_source_counts[REPLAY_SOURCE_ONLINE],
            "current_raw_samples": len(all_samples),
            "current_train_samples": len(current_train_samples),
            "current_admitted_samples": len(admitted_samples),
            "train_dataset_size": train_dataset_size,
            "target_mean": normalization.mean,
            "target_std": normalization.std,
            **train_stats,
        }
        if round_index % config.checkpoint_interval == 0:
            _save_checkpoint(
                last_checkpoint,
                model=model,
                optimizer=optimizer,
                config=config,
                normalization=normalization,
                round_index=round_index,
                stats=row,
            )

        metric: float | None = None
        metric_name: str | None = None
        if (
            config.eval_interval > 0
            and config.eval_episodes > 0
            and round_index % config.eval_interval == 0
        ):
            _save_checkpoint(
                last_checkpoint,
                model=model,
                optimizer=optimizer,
                config=config,
                normalization=normalization,
                round_index=round_index,
                stats=row,
            )
            eval_summary = _evaluate_round(
                config=config,
                checkpoint_path=last_checkpoint,
                round_index=round_index,
                device=device,
            )
            row["evaluation"] = eval_summary
            metric = float(eval_summary["mean_score"])
            metric_name = "evaluation_mean_score"
        elif config.eval_interval == 0 or config.eval_episodes <= 0:
            metric = float(np.mean(episode_scores)) if episode_scores else 0.0
            metric_name = "episode_score_mean"

        if metric is not None:
            row["selection_metric"] = metric
            row["selection_metric_name"] = metric_name

        if metric is not None and metric >= best_metric:
            best_metric = metric
            best_round = round_index
            _save_checkpoint(
                best_checkpoint,
                model=model,
                optimizer=optimizer,
                config=config,
                normalization=normalization,
                round_index=round_index,
                stats=row,
            )

        history.append(row)
        _write_json(out_dir / "history.json", history)

    if not last_checkpoint.exists():
        _save_checkpoint(
            last_checkpoint,
            model=model,
            optimizer=optimizer,
            config=config,
            normalization=normalization,
            round_index=config.rounds,
            stats=history[-1],
        )

    if best_round is None:
        best_metric = float(history[-1].get("episode_score_mean", history[-1]["episode_score"]))
        _save_checkpoint(
            best_checkpoint,
            model=model,
            optimizer=optimizer,
            config=config,
            normalization=normalization,
            round_index=config.rounds,
            stats=history[-1],
        )

    return {
        "model": model,
        "history": history,
        "best_metric": best_metric,
        "best_round": best_round if best_round is not None else config.rounds,
        "best_checkpoint": str(best_checkpoint),
        "last_checkpoint": str(last_checkpoint),
        "device": str(device),
        "output_dir": str(out_dir),
        "target_mean": normalization.mean,
        "target_std": normalization.std,
    }
