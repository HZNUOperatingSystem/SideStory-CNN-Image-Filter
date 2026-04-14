from dataclasses import dataclass
from pathlib import Path

import torch

from .checkpoint import load_model_checkpoint
from .checkpoint_resolution import (
    CheckpointCommandPolicy,
    resolve_checkpoint_command,
)
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
    resolved_command = resolve_checkpoint_command(
        run_dir=config.run_dir,
        ckpt=config.ckpt,
        output=config.output,
        policy=CheckpointCommandPolicy(
            default_output=(
                lambda run_dir: run_dir / f'model.{config.precision}.onnx'
            ),
            output_conflict_message=(
                'Do not set --output when using a run directory.'
            ),
            output_required_message=(
                '--output is required when using --ckpt.'
            ),
        ),
    )
    return OnnxExportConfig(
        run_dir=None,
        ckpt=resolved_command.checkpoint_path,
        output=resolved_command.output_path,
        precision=config.precision,
        height=config.height,
        width=config.width,
        opset=config.opset,
    )


def load_export_checkpoint(config: OnnxExportConfig) -> LoadedExportCheckpoint:
    resolved_config = resolve_onnx_export_config(config)
    checkpoint_path = resolved_config.ckpt
    if checkpoint_path is None:
        msg = 'Checkpoint path could not be resolved.'
        raise ValueError(msg)
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
