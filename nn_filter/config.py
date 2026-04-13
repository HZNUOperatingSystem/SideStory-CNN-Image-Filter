from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

ColorMode = Literal['rgb', 'y-only']
StatusSelection = bool | list[str]


# MARK: - configs


@dataclass(slots=True)
class TrainConfig:
    train_manifest: Path
    val_manifest: Path
    color_mode: ColorMode = 'rgb'
    patch_size: int | None = None
    status: StatusSelection = False
    target_value: str | None = None
    watched_best: list[str] = field(default_factory=list)
    seed: int = 42
    epochs: int = 10
    batch_size: int = 8
    lr: float = 1e-3
    lr_min: float = 1e-6
    runs_dir: Path = Path('runs')
    num_workers: int = 2


@dataclass(slots=True)
class InferConfig:
    run_dir: Path | None = None
    ckpt: Path | None = None
    input: Path = Path()
    output: Path | None = None


# MARK: - helpers


def color_mode_channels(color_mode: ColorMode) -> int:
    return 3 if color_mode == 'rgb' else 1
