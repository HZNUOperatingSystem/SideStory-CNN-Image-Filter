import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x + residual


class CNNFilter(nn.Module):
    def __init__(
        self, in_channels: int = 3, base_channels: int = 64, num_blocks: int = 4
    ):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.blocks = nn.Sequential(
            *[ResidualBlock(base_channels) for _ in range(num_blocks)]
        )
        self.output_conv = nn.Conv2d(base_channels, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.input_conv(x)
        x = self.blocks(x)
        x = self.output_conv(x)
        return x + residual
