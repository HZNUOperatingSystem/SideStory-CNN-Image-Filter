from dataclasses import dataclass
from pathlib import Path

import torch

from .checkpoint import load_model_checkpoint, resolve_run_checkpoint_path
from .config import (
    ColorMode,
    ExportPrecision,
    OnnxExportConfig,
)


@dataclass(frozen=True, slots=True)
class LoadedExportCheckpoint:
    checkpoint_path: Path
    output_path: Path
    precision: ExportPrecision
    color_mode: ColorMode
    model: torch.nn.Module


def resolve_onnx_export_config(
    config: OnnxExportConfig,
) -> OnnxExportConfig:
    if config.run_dir is not None:
        if config.output is not None:
            msg = 'Do not set --output when using a run directory.'
            raise ValueError(msg)
        return OnnxExportConfig(
            run_dir=config.run_dir,
            ckpt=None,
            output=config.run_dir / f'model.{config.precision}.onnx',
            precision=config.precision,
            height=config.height,
            width=config.width,
            opset=config.opset,
        )

    if config.output is None:
        msg = '--output is required when using --ckpt.'
        raise ValueError(msg)
    return config


def load_export_checkpoint(config: OnnxExportConfig) -> LoadedExportCheckpoint:
    resolved_config = resolve_onnx_export_config(config)
    checkpoint_path = resolve_run_checkpoint_path(
        run_dir=resolved_config.run_dir,
        ckpt=resolved_config.ckpt,
    )
    loaded_checkpoint = load_model_checkpoint(
        checkpoint_path,
        device=torch.device('cpu'),
    )

    output_path = resolved_config.output
    if output_path is None:
        msg = 'Output path could not be resolved.'
        raise ValueError(msg)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    return LoadedExportCheckpoint(
        checkpoint_path=loaded_checkpoint.checkpoint_path,
        output_path=output_path,
        precision=resolved_config.precision,
        color_mode=loaded_checkpoint.color_mode,
        model=loaded_checkpoint.model,
    )


def export_dtype(precision: ExportPrecision) -> torch.dtype:
    if precision in {'fp32', 'int8'}:
        return torch.float32
    if precision == 'fp16':
        return torch.float16
    return torch.bfloat16
