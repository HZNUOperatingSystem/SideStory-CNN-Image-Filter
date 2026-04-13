import torch
from torch import nn


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(
            module.weight,
            mode='fan_out',
            nonlinearity='relu',
        )
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(channels)
        self._init_residual_scale()

    def _init_residual_scale(self) -> None:
        nn.init.zeros_(self.conv2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.gelu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return x + residual


class CNNFilter(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_blocks: int = 8,
    ) -> None:
        super().__init__()
        self.input_conv = nn.Conv2d(
            in_channels,
            base_channels,
            kernel_size=3,
            padding=1,
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(base_channels) for _ in range(num_blocks)]
        )
        self.output_conv = nn.Conv2d(
            base_channels,
            in_channels,
            kernel_size=3,
            padding=1,
        )
        self.apply(_init_weights)
        nn.init.zeros_(self.output_conv.weight)
        if self.output_conv.bias is not None:
            nn.init.zeros_(self.output_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.input_conv(x)
        x = self.blocks(x)
        x = self.output_conv(x)
        return torch.clamp(x + residual, 0.0, 1.0)
