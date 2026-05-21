from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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
) -> float:
    model.train(optimizer is not None)
    losses: list[float] = []

    for boards, legal_masks, target_probs, weights in loader:
        non_blocking = device.type == "cuda"
        boards = boards.to(device, non_blocking=non_blocking)
        legal_masks = legal_masks.to(device, non_blocking=non_blocking)
        target_probs = target_probs.to(device, non_blocking=non_blocking)
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

    return float(np.mean(losses)) if losses else 0.0


def train_supervised_cnn(config: SupervisedTrainingConfig) -> dict[str, Any]:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = resolve_device(config.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "training_config.json").write_text(
        json.dumps(asdict(config), indent=2, sort_keys=True),
        encoding="utf-8",
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
        train_loss = _run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            temperature=config.temperature,
            optimizer=optimizer,
        )
        with torch.no_grad():
            validation_loss = _run_epoch(
                model=model,
                loader=val_loader,
                device=device,
                temperature=config.temperature,
                optimizer=None,
            )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
        _save_checkpoint(
            path=out_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            validation_loss=validation_loss,
            training_config=config,
        )
        if validation_loss <= best_validation_loss:
            best_validation_loss = validation_loss
            _save_checkpoint(
                path=out_dir / "best.pt",
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
    return {
        "history": history,
        "best_validation_loss": best_validation_loss,
        "best_checkpoint": str(out_dir / "best.pt"),
        "last_checkpoint": str(out_dir / "last.pt"),
        "device": str(device),
    }
