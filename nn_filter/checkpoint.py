from dataclasses import dataclass
from pathlib import Path

import torch

from .config import ColorMode, color_mode_channels
from .model import CNNFilter


@dataclass(frozen=True, slots=True)
class LoadedModelCheckpoint:
    checkpoint_path: Path
    color_mode: ColorMode
    model: CNNFilter


def require_checkpoint_path(checkpoint_path: Path | None) -> Path:
    if checkpoint_path is None:
        msg = 'Checkpoint path is required.'
        raise ValueError(msg)
    if not checkpoint_path.is_file():
        msg = f'Checkpoint not found: {checkpoint_path}'
        raise FileNotFoundError(msg)
    return checkpoint_path


def resolve_run_checkpoint_path(
    *,
    run_dir: Path | None,
    ckpt: Path | None,
) -> Path:
    if (run_dir is None) == (ckpt is None):
        msg = 'Provide either a run directory or --ckpt.'
        raise ValueError(msg)
    if run_dir is not None:
        return require_checkpoint_path(run_dir / 'best.pt')
    return require_checkpoint_path(ckpt)


def load_model_checkpoint(
    checkpoint_path: Path,
    *,
    device: torch.device | str,
) -> LoadedModelCheckpoint:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    raw_model_config = checkpoint.get('model_config')
    if not isinstance(raw_model_config, dict):
        msg = f'Checkpoint {checkpoint_path} is missing model_config.'
        raise ValueError(msg)
    color_mode = raw_model_config.get('color_mode')
    if color_mode not in {'rgb', 'y-only'}:
        msg = (
            'Unsupported color_mode in checkpoint '
            f'{checkpoint_path}: {color_mode!r}'
        )
        raise ValueError(msg)

    state_dict = checkpoint.get('model_state_dict')
    if not isinstance(state_dict, dict):
        msg = f'Checkpoint {checkpoint_path} is missing model_state_dict.'
        raise ValueError(msg)

    model = CNNFilter(in_channels=color_mode_channels(color_mode)).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return LoadedModelCheckpoint(
        checkpoint_path=checkpoint_path,
        color_mode=color_mode,
        model=model,
    )
