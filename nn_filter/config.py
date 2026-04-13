from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ColorMode = Literal['rgb', 'y-only']


# MARK: - configs


@dataclass(slots=True)
class TrainConfig:
    train_manifest: Path
    val_manifest: Path
    color_mode: ColorMode = 'rgb'
    patch_size: int | None = None
    epochs: int = 10
    batch_size: int = 8
    lr: float = 1e-3
    save_dir: Path = Path('checkpoints')
    num_workers: int = 2


# MARK: - helpers


def color_mode_channels(color_mode: ColorMode) -> int:
    return 3 if color_mode == 'rgb' else 1
