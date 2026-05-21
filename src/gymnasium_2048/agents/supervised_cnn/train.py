from __future__ import annotations

import csv
import json
import shutil
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


def _run_epoch(
    model: SupervisedCNN,
    loader: DataLoader,
    device: torch.device,
    temperature: float,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    model.train(optimizer is not None)
    losses: list[float] = []
    total_examples = 0
    top1_correct = 0
    top2_correct = 0

    for boards, legal_masks, target_probs, actions, weights in loader:
        non_blocking = device.type == "cuda"
        boards = boards.to(device, non_blocking=non_blocking)
        legal_masks = legal_masks.to(device, non_blocking=non_blocking)
        target_probs = target_probs.to(device, non_blocking=non_blocking)
        actions = actions.to(device, non_blocking=non_blocking)
        weights = weights.to(device, non_blocking=non_blocking)

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        logits = model(boards)
        loss = masked_soft_cross_entropy(
            logits=logits,
            target_probs=target_probs,
            legal_mask=legal_masks,
            sample_weight=weights,
            temperature=temperature,
        )

        if optimizer is not None:
            loss.backward()
            optimizer.step()

        losses.append(float(loss.detach().cpu()))
        with torch.no_grad():
            masked_logits = logits.masked_fill(~legal_masks.bool(), -1.0e9)
            top2 = torch.topk(masked_logits, k=2, dim=-1).indices
            total_examples += int(actions.numel())
            top1_correct += int((top2[:, 0] == actions).sum().detach().cpu())
            top2_correct += int((top2 == actions.unsqueeze(-1)).any(dim=-1).sum().detach().cpu())

    if total_examples == 0:
        return {"loss": 0.0, "top1_accuracy": 0.0, "top2_accuracy": 0.0}
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "top1_accuracy": top1_correct / total_examples,
        "top2_accuracy": top2_correct / total_examples,
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


def train_supervised_cnn(config: SupervisedTrainingConfig) -> dict[str, Any]:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = resolve_device(config.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

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
    train_dataset = ExpectimaxDataset(
        path=config.data_path,
        indices=train_indices,
        weight_config=weight_config,
    )
    val_dataset = ExpectimaxDataset(
        path=config.data_path,
        indices=val_indices,
        weight_config=weight_config,
    )

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )

    model = SupervisedCNN(CNNConfig()).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

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
        )
        with torch.no_grad():
            validation_metrics = _run_epoch(
                model=model,
                loader=val_loader,
                device=device,
                temperature=config.temperature,
                optimizer=None,
            )
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
            }
        )
        _save_checkpoint(
            path=artifact_dirs["checkpoints"] / "last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            validation_loss=validation_loss,
            training_config=config,
        )
        if validation_loss <= best_validation_loss:
            best_validation_loss = validation_loss
            _save_checkpoint(
                path=artifact_dirs["checkpoints"] / "best.pt",
                model=model,
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
