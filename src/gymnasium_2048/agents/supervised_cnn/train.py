from __future__ import annotations

import csv
import json
import shutil
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import trange

from gymnasium_2048.agents.supervised_cnn.data import (
    ExpectimaxDataset,
    WeightConfig,
    split_indices,
)
from gymnasium_2048.agents.supervised_cnn.encoding import encode_boards_torch
from gymnasium_2048.agents.supervised_cnn.loss import masked_soft_cross_entropy
from gymnasium_2048.agents.supervised_cnn.model import (
    CNNConfig,
    SupervisedCNN,
    config_to_dict,
)


@dataclass(frozen=True)
class SupervisedTrainingConfig:
    data_path: str
    out_dir: str
    epochs: int = 1
    batch_size: int = 64
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    temperature: float = 1.0
    validation_fraction: float = 0.2
    device: str = "auto"
    seed: int = 42
    use_high_score_weighting: bool = True
    score_weight: float = 0.5
    tile_weight: float = 0.4
    late_game_weight: float = 0.2
    difficulty_weight: float = 0.2
    max_sample_weight: float = 3.0
    copy_training_data: bool = True
    num_workers: int = 4
    persistent_workers: bool = True
    prefetch_factor: int | None = 4
    pin_memory: bool = True
    drop_last: bool = False
    encode_on_device: bool = True
    compute_train_accuracy: bool = False
    validation_interval: int = 1
    validation_max_samples: int | None = 100_000
    amp: bool = True
    allow_tf32: bool = True
    torch_compile: bool = False
    profile: bool = True
    profile_batches: int = 50


def resolve_device(device: str) -> torch.device:
    requested = device.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is False. "
            "Use --device cpu or install a CUDA-enabled PyTorch build."
        )
    return torch.device(device)


def _save_checkpoint(
    path: Path,
    model: SupervisedCNN,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation_loss: float,
    training_config: SupervisedTrainingConfig,
) -> None:
    payload = {
        "model_config": config_to_dict(model.config),
        "training_config": asdict(training_config),
        "epoch": epoch,
        "validation_loss": validation_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(payload, path)


def _make_grad_scaler(enabled: bool) -> torch.amp.GradScaler:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast_context(device: torch.device, enabled: bool):
    if enabled:
        return torch.amp.autocast(device_type=device.type)
    return nullcontext()


def _sync_if_profile(device: torch.device, enabled: bool) -> None:
    if enabled and device.type == "cuda":
        torch.cuda.synchronize(device)


def _prepare_boards(
    boards: torch.Tensor,
    device: torch.device,
    input_channels: int,
    encode_on_device: bool,
    non_blocking: bool,
) -> torch.Tensor:
    if not encode_on_device:
        return boards.to(device, non_blocking=non_blocking)

    boards = boards.to(device, dtype=torch.long, non_blocking=non_blocking)
    return encode_boards_torch(boards, num_channels=input_channels)


def _run_epoch(
    model: SupervisedCNN,
    loader: DataLoader,
    device: torch.device,
    temperature: float,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    amp_enabled: bool = False,
    encode_on_device: bool = False,
    input_channels: int = 16,
    compute_accuracy: bool = True,
    profile: bool = False,
    profile_batches: int = 50,
) -> dict[str, float]:
    model.train(optimizer is not None)
    total_examples = 0
    total_loss = torch.zeros((), device=device)
    top1_correct = torch.zeros((), device=device)
    top2_correct = torch.zeros((), device=device)
    data_wait_seconds = 0.0
    transfer_seconds = 0.0
    forward_loss_seconds = 0.0
    backward_seconds = 0.0
    profile_batch_count = 0
    start_time = time.perf_counter()

    iterator = iter(loader)
    batch_index = 0
    while True:
        data_start = time.perf_counter()
        try:
            boards, legal_masks, target_probs, actions, weights = next(iterator)
        except StopIteration:
            break
        data_wait_seconds += time.perf_counter() - data_start

        non_blocking = device.type == "cuda"
        should_profile_batch = profile and (
            profile_batches <= 0 or batch_index < profile_batches
        )

        _sync_if_profile(device, should_profile_batch)
        transfer_start = time.perf_counter()
        batch_size = int(actions.numel())
        boards = _prepare_boards(
            boards=boards,
            device=device,
            input_channels=input_channels,
            encode_on_device=encode_on_device,
            non_blocking=non_blocking,
        )
        legal_masks = legal_masks.to(device, non_blocking=non_blocking)
        target_probs = target_probs.to(device, non_blocking=non_blocking)
        if compute_accuracy:
            actions = actions.to(device, non_blocking=non_blocking)
        weights = weights.to(device, non_blocking=non_blocking)
        _sync_if_profile(device, should_profile_batch)
        transfer_seconds += time.perf_counter() - transfer_start

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        _sync_if_profile(device, should_profile_batch)
        forward_start = time.perf_counter()
        with _autocast_context(device, amp_enabled):
            logits = model(boards)
            loss = masked_soft_cross_entropy(
                logits=logits,
                target_probs=target_probs,
                legal_mask=legal_masks,
                sample_weight=weights,
                temperature=temperature,
            )
        _sync_if_profile(device, should_profile_batch)
        forward_loss_seconds += time.perf_counter() - forward_start

        if optimizer is not None:
            _sync_if_profile(device, should_profile_batch)
            backward_start = time.perf_counter()
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            _sync_if_profile(device, should_profile_batch)
            backward_seconds += time.perf_counter() - backward_start

        total_examples += batch_size
        total_loss = total_loss + loss.detach() * batch_size
        with torch.no_grad():
            if compute_accuracy:
                masked_logits = logits.float().masked_fill(
                    ~legal_masks.bool(),
                    -1.0e9,
                )
                top2 = torch.topk(masked_logits, k=2, dim=-1).indices
                top1_correct = top1_correct + (top2[:, 0] == actions).sum()
                top2_correct = top2_correct + (
                    top2 == actions.unsqueeze(-1)
                ).any(dim=-1).sum()

        if should_profile_batch:
            profile_batch_count += 1
        batch_index += 1

    if total_examples == 0:
        return {
            "loss": 0.0,
            "top1_accuracy": 0.0,
            "top2_accuracy": 0.0,
            "samples_per_second": 0.0,
            "data_wait_seconds": 0.0,
            "transfer_seconds": 0.0,
            "forward_loss_seconds": 0.0,
            "backward_seconds": 0.0,
            "profiled_batches": 0.0,
        }

    total_seconds = time.perf_counter() - start_time
    loss_value = float((total_loss / total_examples).detach().cpu())
    if compute_accuracy:
        top1_accuracy = float((top1_correct / total_examples).detach().cpu())
        top2_accuracy = float((top2_correct / total_examples).detach().cpu())
    else:
        top1_accuracy = float("nan")
        top2_accuracy = float("nan")

    return {
        "loss": loss_value,
        "top1_accuracy": top1_accuracy,
        "top2_accuracy": top2_accuracy,
        "samples_per_second": total_examples / max(total_seconds, 1.0e-9),
        "data_wait_seconds": data_wait_seconds,
        "transfer_seconds": transfer_seconds,
        "forward_loss_seconds": forward_loss_seconds,
        "backward_seconds": backward_seconds,
        "profiled_batches": float(profile_batch_count),
    }


def _write_history_csv(history: list[dict[str, float | int]], path: Path) -> None:
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def _summarize_split(
    name: str,
    dataset: ExpectimaxDataset,
    indices: np.ndarray,
) -> dict[str, float | int | str]:
    final_scores = dataset.final_scores[indices]
    final_max_tiles = dataset.final_max_tiles[indices]
    empty_counts = dataset.empty_counts[indices]
    weights = dataset.weights[indices]
    return {
        "split": name,
        "samples": int(len(indices)),
        "mean_final_score": float(np.mean(final_scores)),
        "median_final_score": float(np.median(final_scores)),
        "max_final_score": int(np.max(final_scores)),
        "mean_final_max_tile": float(np.mean(final_max_tiles)),
        "max_final_max_tile": int(np.max(final_max_tiles)),
        "reach_2048_rate": float(np.mean(final_max_tiles >= 2048)),
        "mean_empty_cells": float(np.mean(empty_counts)),
        "mean_sample_weight": float(np.mean(weights)),
        "max_sample_weight": float(np.max(weights)),
    }


def _write_split_summary(summary: list[dict[str, float | int | str]], out_dir: Path) -> None:
    (out_dir / "dataset_split_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    with (out_dir / "dataset_split_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)


def _make_artifact_dirs(out_dir: Path) -> dict[str, Path]:
    dirs = {
        "checkpoints": out_dir / "checkpoints",
        "plots": out_dir / "plots",
        "data": out_dir / "data",
        "tables": out_dir / "tables",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def _copy_training_data(data_path: str, data_dir: Path) -> str | None:
    source = Path(data_path)
    if not source.exists():
        return None
    destination = data_dir / source.name
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    return str(destination)


def _plot_training_curves(history: list[dict[str, float | int]], plots_dir: Path) -> None:
    if not history:
        return
    epochs = [int(row["epoch"]) for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, [float(row["train_loss"]) for row in history], label="train")
    axes[0].plot(epochs, [float(row["validation_loss"]) for row in history], label="validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Masked soft CE")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(epochs, [float(row["train_top1_accuracy"]) for row in history], label="train top-1")
    axes[1].plot(epochs, [float(row["validation_top1_accuracy"]) for row in history], label="val top-1")
    axes[1].plot(epochs, [float(row["train_top2_accuracy"]) for row in history], linestyle="--", label="train top-2")
    axes[1].plot(epochs, [float(row["validation_top2_accuracy"]) for row in history], linestyle="--", label="val top-2")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_title("Teacher Action Match")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(plots_dir / "training_curves.png", dpi=150)
    plt.close(fig)


def _plot_dataset_analysis(
    dataset: ExpectimaxDataset,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    plots_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    split_data = (("train", train_indices), ("validation", val_indices))

    for label, indices in split_data:
        axes[0, 0].hist(dataset.final_scores[indices], bins=40, alpha=0.55, label=label)
    axes[0, 0].set_title("Final Score Distribution")
    axes[0, 0].set_xlabel("Final score")
    axes[0, 0].set_ylabel("Samples")
    axes[0, 0].legend()

    all_tiles = sorted(set(dataset.final_max_tiles[train_indices].tolist()) | set(dataset.final_max_tiles[val_indices].tolist()))
    x = np.arange(len(all_tiles))
    width = 0.38
    for offset, (label, indices) in zip((-width / 2, width / 2), split_data):
        counts = np.asarray([np.count_nonzero(dataset.final_max_tiles[indices] == tile) for tile in all_tiles])
        axes[0, 1].bar(x + offset, counts, width=width, label=label)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels([str(tile) for tile in all_tiles], rotation=30)
    axes[0, 1].set_title("Final Max Tile")
    axes[0, 1].set_ylabel("Samples")
    axes[0, 1].legend()

    for label, indices in split_data:
        axes[1, 0].hist(dataset.empty_counts[indices], bins=np.arange(18) - 0.5, alpha=0.55, label=label)
    axes[1, 0].set_title("Empty Cells")
    axes[1, 0].set_xlabel("Empty cells before action")
    axes[1, 0].set_ylabel("Samples")
    axes[1, 0].legend()

    for label, indices in split_data:
        axes[1, 1].hist(dataset.weights[indices], bins=40, alpha=0.55, label=label)
    axes[1, 1].set_title("Sample Weights")
    axes[1, 1].set_xlabel("Weight")
    axes[1, 1].set_ylabel("Samples")
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(plots_dir / "dataset_analysis.png", dpi=150)
    plt.close(fig)


def _limit_indices(
    indices: np.ndarray,
    max_samples: int | None,
    seed: int,
) -> np.ndarray:
    if max_samples is None or max_samples <= 0 or len(indices) <= max_samples:
        return indices
    rng = np.random.default_rng(seed)
    selected = rng.choice(indices, size=max_samples, replace=False)
    return np.asarray(selected, dtype=np.int64)


def _seed_worker(worker_id: int) -> None:
    np.random.seed((torch.initial_seed() + worker_id) % 2**32)


def _make_loader(
    dataset: ExpectimaxDataset,
    config: SupervisedTrainingConfig,
    shuffle: bool,
    pin_memory: bool,
    drop_last: bool = False,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    num_workers = max(int(config.num_workers), 0)
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": config.batch_size,
        "shuffle": shuffle,
        "pin_memory": pin_memory,
        "num_workers": num_workers,
        "drop_last": drop_last,
        "generator": generator,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = config.persistent_workers
        kwargs["worker_init_fn"] = _seed_worker
        if config.prefetch_factor is not None:
            kwargs["prefetch_factor"] = config.prefetch_factor
    return DataLoader(**kwargs)


def train_supervised_cnn(config: SupervisedTrainingConfig) -> dict[str, Any]:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = resolve_device(config.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32
        torch.backends.cudnn.allow_tf32 = config.allow_tf32

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_dirs = _make_artifact_dirs(out_dir)
    (out_dir / "training_config.json").write_text(
        json.dumps(asdict(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    copied_data_path = (
        _copy_training_data(config.data_path, artifact_dirs["data"])
        if config.copy_training_data
        else None
    )

    weight_config = WeightConfig(
        enabled=config.use_high_score_weighting,
        score_weight=config.score_weight,
        tile_weight=config.tile_weight,
        late_game_weight=config.late_game_weight,
        difficulty_weight=config.difficulty_weight,
        max_weight=config.max_sample_weight,
    )

    base_dataset = ExpectimaxDataset(
        path=config.data_path,
        weight_config=weight_config,
    )
    train_indices, val_indices = split_indices(
        n_samples=len(base_dataset.boards),
        validation_fraction=config.validation_fraction,
        seed=config.seed,
    )
    split_summary = [
        _summarize_split("train", base_dataset, train_indices),
        _summarize_split("validation", base_dataset, val_indices),
    ]
    _write_split_summary(split_summary, out_dir)
    _write_split_summary(split_summary, artifact_dirs["tables"])
    np.save(artifact_dirs["data"] / "train_indices.npy", train_indices)
    np.save(artifact_dirs["data"] / "validation_indices.npy", val_indices)
    _plot_dataset_analysis(base_dataset, train_indices, val_indices, artifact_dirs["plots"])
    val_loader_indices = _limit_indices(
        indices=val_indices,
        max_samples=config.validation_max_samples,
        seed=config.seed + 1,
    )
    train_dataset = base_dataset.subset(
        train_indices,
        encode_boards=not config.encode_on_device,
    )
    val_dataset = base_dataset.subset(
        val_loader_indices,
        encode_boards=not config.encode_on_device,
    )

    pin_memory = bool(config.pin_memory and device.type == "cuda")
    train_loader = _make_loader(
        dataset=train_dataset,
        config=config,
        shuffle=True,
        pin_memory=pin_memory,
        drop_last=config.drop_last,
    )
    val_loader = _make_loader(
        dataset=val_dataset,
        config=config,
        shuffle=False,
        pin_memory=pin_memory,
        drop_last=False,
    )

    raw_model = SupervisedCNN(CNNConfig()).to(device)
    model = raw_model
    if config.torch_compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch_compile=True requires a PyTorch build with torch.compile")
        model = torch.compile(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    amp_enabled = bool(config.amp and device.type == "cuda")
    scaler = _make_grad_scaler(amp_enabled) if amp_enabled else None

    best_validation_loss = float("inf")
    history: list[dict[str, float | int]] = []

    print(f"Supervised CNN training device: {device}")

    for epoch in trange(1, config.epochs + 1, desc="Supervised train", unit="epoch"):
        train_metrics = _run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            temperature=config.temperature,
            optimizer=optimizer,
            scaler=scaler,
            amp_enabled=amp_enabled,
            encode_on_device=config.encode_on_device,
            input_channels=raw_model.config.input_channels,
            compute_accuracy=config.compute_train_accuracy,
            profile=config.profile,
            profile_batches=config.profile_batches,
        )
        should_validate = (
            config.validation_interval <= 1
            or epoch % config.validation_interval == 0
            or epoch == config.epochs
        )
        if should_validate:
            validation_start = time.perf_counter()
            with torch.no_grad():
                validation_metrics = _run_epoch(
                    model=model,
                    loader=val_loader,
                    device=device,
                    temperature=config.temperature,
                    optimizer=None,
                    scaler=None,
                    amp_enabled=amp_enabled,
                    encode_on_device=config.encode_on_device,
                    input_channels=raw_model.config.input_channels,
                    compute_accuracy=True,
                    profile=config.profile,
                    profile_batches=config.profile_batches,
                )
            validation_seconds = time.perf_counter() - validation_start
        else:
            validation_metrics = {
                "loss": float("nan"),
                "top1_accuracy": float("nan"),
                "top2_accuracy": float("nan"),
                "samples_per_second": 0.0,
                "data_wait_seconds": 0.0,
                "transfer_seconds": 0.0,
                "forward_loss_seconds": 0.0,
                "backward_seconds": 0.0,
                "profiled_batches": 0.0,
            }
            validation_seconds = 0.0
        train_loss = train_metrics["loss"]
        validation_loss = validation_metrics["loss"]

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "train_top1_accuracy": train_metrics["top1_accuracy"],
                "validation_top1_accuracy": validation_metrics["top1_accuracy"],
                "train_top2_accuracy": train_metrics["top2_accuracy"],
                "validation_top2_accuracy": validation_metrics["top2_accuracy"],
                "train_samples_per_second": train_metrics["samples_per_second"],
                "validation_samples_per_second": validation_metrics["samples_per_second"],
                "train_data_wait_seconds": train_metrics["data_wait_seconds"],
                "train_transfer_seconds": train_metrics["transfer_seconds"],
                "train_forward_loss_seconds": train_metrics["forward_loss_seconds"],
                "train_backward_seconds": train_metrics["backward_seconds"],
                "validation_data_wait_seconds": validation_metrics["data_wait_seconds"],
                "validation_transfer_seconds": validation_metrics["transfer_seconds"],
                "validation_forward_loss_seconds": validation_metrics["forward_loss_seconds"],
                "validation_seconds": validation_seconds,
                "validation_samples": int(len(val_loader_indices)),
                "profiled_batches": train_metrics["profiled_batches"],
            }
        )
        _save_checkpoint(
            path=artifact_dirs["checkpoints"] / "last.pt",
            model=raw_model,
            optimizer=optimizer,
            epoch=epoch,
            validation_loss=validation_loss,
            training_config=config,
        )
        if should_validate and validation_loss <= best_validation_loss:
            best_validation_loss = validation_loss
            _save_checkpoint(
                path=artifact_dirs["checkpoints"] / "best.pt",
                model=raw_model,
                optimizer=optimizer,
                epoch=epoch,
                validation_loss=validation_loss,
                training_config=config,
            )

    (out_dir / "history.json").write_text(
        json.dumps(history, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (artifact_dirs["tables"] / "history.json").write_text(
        json.dumps(history, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_history_csv(history, artifact_dirs["tables"] / "history.csv")
    _plot_training_curves(history, artifact_dirs["plots"])
    return {
        "history": history,
        "best_validation_loss": best_validation_loss,
        "best_checkpoint": str(artifact_dirs["checkpoints"] / "best.pt"),
        "last_checkpoint": str(artifact_dirs["checkpoints"] / "last.pt"),
        "device": str(device),
        "output_dir": str(out_dir),
        "copied_data_path": copied_data_path,
    }
