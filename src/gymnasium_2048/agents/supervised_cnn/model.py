from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class CNNConfig:
    input_channels: int = 16
    conv_channels: int = 64
    hidden_size: int = 128


class SupervisedCNN(nn.Module):
    """No-pooling CNN that regresses one afterstate continuation value."""

    def __init__(self, config: CNNConfig | None = None) -> None:
        super().__init__()
        self.config = config or CNNConfig()
        self.features = nn.Sequential(
            nn.Conv2d(
                self.config.input_channels,
                32,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                32,
                self.config.conv_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.config.conv_channels * 4 * 4, self.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size, 1),
        )

    def forward(self, boards: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.features(boards)).squeeze(-1)


def config_to_dict(config: CNNConfig) -> dict[str, int]:
    return asdict(config)


def config_from_dict(data: dict[str, int] | None) -> CNNConfig:
    return CNNConfig(**data) if data else CNNConfig()
