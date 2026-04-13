from dataclasses import dataclass
from pathlib import Path

import torch

from .config import (
    ColorMode,
    ExportPrecision,
    OnnxExportConfig,
    color_mode_channels,
)
from .model import CNNFilter


@dataclass(frozen=True, slots=True)
class LoadedExportCheckpoint:
    checkpoint_path: Path
    output_path: Path
    precision: ExportPrecision
    color_mode: ColorMode
    model: CNNFilter


def resolve_onnx_export_config(
    config: OnnxExportConfig,
) -> OnnxExportConfig:
    if (config.run_dir is None) == (config.ckpt is None):
        msg = 'Provide either a run directory or --ckpt.'
        raise ValueError(msg)

    if config.run_dir is not None:
        if config.output is not None:
            msg = 'Do not set --output when using a run directory.'
            raise ValueError(msg)
        return OnnxExportConfig(
            run_dir=config.run_dir,
            ckpt=config.run_dir / 'best.pt',
            output=config.run_dir / f'model.{config.precision}.onnx',
            precision=config.precision,
            height=config.height,
            width=config.width,
            opset=config.opset,
        )

    if config.ckpt is None:
        msg = 'Checkpoint path is required.'
        raise ValueError(msg)
    if config.output is None:
        msg = '--output is required when using --ckpt.'
        raise ValueError(msg)
    return config


def load_export_checkpoint(config: OnnxExportConfig) -> LoadedExportCheckpoint:
    resolved_config = resolve_onnx_export_config(config)
    checkpoint_path = _require_checkpoint_path(resolved_config.ckpt)
    checkpoint = torch.load(
        checkpoint_path,
        map_location='cpu',
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

    model = CNNFilter(in_channels=color_mode_channels(color_mode))
    state_dict = checkpoint.get('model_state_dict')
    if not isinstance(state_dict, dict):
        msg = f'Checkpoint {checkpoint_path} is missing model_state_dict.'
        raise ValueError(msg)
    model.load_state_dict(state_dict)
    model.eval()

    output_path = resolved_config.output
    if output_path is None:
        msg = 'Output path could not be resolved.'
        raise ValueError(msg)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    return LoadedExportCheckpoint(
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        precision=resolved_config.precision,
        color_mode=color_mode,
        model=model,
    )


def export_dtype(precision: ExportPrecision) -> torch.dtype:
    if precision in {'fp32', 'int8'}:
        return torch.float32
    if precision == 'fp16':
        return torch.float16
    return torch.bfloat16


def _require_checkpoint_path(checkpoint_path: Path | None) -> Path:
    if checkpoint_path is None:
        msg = 'Checkpoint path is required.'
        raise ValueError(msg)
    if not checkpoint_path.is_file():
        msg = f'Checkpoint not found: {checkpoint_path}'
        raise FileNotFoundError(msg)
    return checkpoint_path
