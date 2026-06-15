from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from gymnasium_2048.agents.supervised_cnn.data import (
    AfterstateDataset,
    split_grouped_indices,
)
from gymnasium_2048.agents.supervised_cnn.loss import regression_loss
from gymnasium_2048.agents.supervised_cnn.model import (
    CNNConfig,
    SupervisedCNN,
    config_to_dict,
)


@dataclass(frozen=True)
class SupervisedTrainingConfig:
    data_path: str
    out_dir: str
    epochs: int = 10
    batch_size: int = 256
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    validation_fraction: float = 0.2
    device: str = "auto"
    seed: int = 42
    num_workers: int = 0
    pin_memory: bool = True
    input_channels: int = 16
    conv_channels: int = 64
    hidden_size: int = 128
    loss: str = "huber"
    target_normalization: bool = True
    symmetry_augmentation: bool = True


def resolve_device(device: str) -> torch.device:
    requested = device.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is False"
        )
    return torch.device(device)


def _run_epoch(
    model: SupervisedCNN,
    loader: DataLoader,
    device: torch.device,
    target_mean: float,
    target_std: float,
    loss_kind: str,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0

    for boards, targets in loader:
        boards = boards.to(device)
        targets = targets.to(device)
        normalized_targets = (targets - target_mean) / target_std
        if training:
            optimizer.zero_grad(set_to_none=True)
        predictions = model(boards)
        loss = regression_loss(predictions, normalized_targets, kind=loss_kind)
        if training:
            loss.backward()
            optimizer.step()
        batch_size = int(len(targets))
        total_loss += float(loss.detach().cpu()) * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


def _save_checkpoint(
    path: Path,
    model: SupervisedCNN,
    optimizer: torch.optim.Optimizer,
    config: SupervisedTrainingConfig,
    target_mean: float,
    target_std: float,
    epoch: int,
    validation_loss: float,
) -> None:
    torch.save(
        {
            "model_config": config_to_dict(model.config),
            "training_config": asdict(config),
            "target_mean": target_mean,
            "target_std": target_std,
            "epoch": epoch,
            "validation_loss": validation_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def train_supervised_cnn(
    config: SupervisedTrainingConfig,
) -> dict[str, Any]:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = resolve_device(config.device)

    base_dataset = AfterstateDataset(
        path=config.data_path,
        num_channels=config.input_channels,
        seed=config.seed,
    )
    train_indices, validation_indices = split_grouped_indices(
        base_dataset.root_ids,
        validation_fraction=config.validation_fraction,
        seed=config.seed,
    )
    train_targets = base_dataset.targets[train_indices]
    target_mean = (
        float(np.mean(train_targets)) if config.target_normalization else 0.0
    )
    target_std = (
        float(np.std(train_targets)) if config.target_normalization else 1.0
    )
    if target_std < 1.0e-6:
        target_std = 1.0

    train_dataset = base_dataset.subset(
        train_indices,
        augment=config.symmetry_augmentation,
        seed=config.seed,
    )
    validation_dataset = base_dataset.subset(
        validation_indices,
        augment=False,
        seed=config.seed + 1,
    )
    pin_memory = bool(config.pin_memory and device.type == "cuda")
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )

    model = SupervisedCNN(
        CNNConfig(
            input_channels=config.input_channels,
            conv_channels=config.conv_channels,
            hidden_size=config.hidden_size,
        )
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    out_dir = Path(config.out_dir)
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "training_config.json").write_text(
        json.dumps(asdict(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    history: list[dict[str, float | int]] = []
    best_validation_loss = float("inf")
    for epoch in range(1, config.epochs + 1):
        train_loss = _run_epoch(
            model,
            train_loader,
            device,
            target_mean,
            target_std,
            config.loss,
            optimizer,
        )
        with torch.no_grad():
            validation_loss = (
                _run_epoch(
                    model,
                    validation_loader,
                    device,
                    target_mean,
                    target_std,
                    config.loss,
                    None,
                )
                if len(validation_dataset) > 0
                else train_loss
            )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
        _save_checkpoint(
            checkpoint_dir / "last.pt",
            model,
            optimizer,
            config,
            target_mean,
            target_std,
            epoch,
            validation_loss,
        )
        if validation_loss <= best_validation_loss:
            best_validation_loss = validation_loss
            _save_checkpoint(
                checkpoint_dir / "best.pt",
                model,
                optimizer,
                config,
                target_mean,
                target_std,
                epoch,
                validation_loss,
            )

    (out_dir / "history.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )
    return {
        "model": model,
        "history": history,
        "best_validation_loss": best_validation_loss,
        "best_checkpoint": str(checkpoint_dir / "best.pt"),
        "last_checkpoint": str(checkpoint_dir / "last.pt"),
        "device": str(device),
        "output_dir": str(out_dir),
        "target_mean": target_mean,
        "target_std": target_std,
    }
