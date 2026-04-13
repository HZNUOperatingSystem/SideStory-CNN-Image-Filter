import csv
from dataclasses import dataclass
from pathlib import Path

import torch

from .checkpoint import load_model_checkpoint, resolve_run_checkpoint_path
from .config import ColorMode, InferConfig
from .io_utils import is_image_path, load_image_tensor


@dataclass(frozen=True, slots=True)
class InferenceSample:
    input_path: Path
    output_path: Path


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    checkpoint_path: Path
    output_dir: Path
    color_mode: ColorMode
    model: torch.nn.Module


def resolve_infer_config(config: InferConfig) -> InferConfig:
    if config.run_dir is not None:
        if config.output is not None:
            msg = 'Do not set --output when using a run directory.'
            raise ValueError(msg)
        return InferConfig(
            run_dir=config.run_dir,
            ckpt=None,
            input=config.input,
            output=config.run_dir / 'outputs',
        )

    if config.output is None:
        msg = '--output is required when using --ckpt.'
        raise ValueError(msg)
    return config


def load_checkpoint(
    config: InferConfig,
    *,
    device: torch.device,
) -> LoadedCheckpoint:
    resolved_config = resolve_infer_config(config)
    checkpoint_path = resolve_run_checkpoint_path(
        run_dir=resolved_config.run_dir,
        ckpt=resolved_config.ckpt,
    )
    loaded_checkpoint = load_model_checkpoint(checkpoint_path, device=device)

    output_dir = resolved_config.output
    if output_dir is None:
        msg = 'Output directory could not be resolved.'
        raise ValueError(msg)
    output_dir.mkdir(parents=True, exist_ok=True)

    return LoadedCheckpoint(
        checkpoint_path=loaded_checkpoint.checkpoint_path,
        output_dir=output_dir,
        color_mode=loaded_checkpoint.color_mode,
        model=loaded_checkpoint.model,
    )


def load_inference_samples(
    input_path: Path,
    *,
    output_dir: Path,
) -> list[InferenceSample]:
    if input_path.is_file() and input_path.suffix.lower() == '.csv':
        return _load_manifest_samples(input_path, output_dir=output_dir)
    if input_path.is_file():
        if not is_image_path(input_path):
            msg = f'Unsupported input file: {input_path}'
            raise ValueError(msg)
        return [
            InferenceSample(
                input_path=input_path,
                output_path=output_dir / input_path.name,
            )
        ]
    if input_path.is_dir():
        image_paths = sorted(
            path
            for path in input_path.rglob('*')
            if path.is_file() and is_image_path(path)
        )
        if not image_paths:
            msg = f'No images found in directory: {input_path}'
            raise ValueError(msg)
        return [
            InferenceSample(
                input_path=image_path,
                output_path=output_dir / image_path.relative_to(input_path),
            )
            for image_path in image_paths
        ]

    msg = f'Input path not found: {input_path}'
    raise FileNotFoundError(msg)


def load_inference_tensor(
    sample: InferenceSample,
    *,
    color_mode: ColorMode,
    device: torch.device,
) -> torch.Tensor:
    return (
        load_image_tensor(
            sample.input_path,
            color_mode=color_mode,
        )
        .unsqueeze(0)
        .to(device)
    )


def _load_manifest_samples(
    manifest_path: Path,
    *,
    output_dir: Path,
) -> list[InferenceSample]:
    samples: list[InferenceSample] = []
    with manifest_path.open(newline='') as manifest_file:
        reader = csv.DictReader(manifest_file)
        fieldnames = reader.fieldnames or []
        required_fields = {'sample', 'kind', 'path'}
        missing_fields = sorted(required_fields - set(fieldnames))
        if missing_fields:
            joined_fields = ', '.join(missing_fields)
            msg = (
                f'Manifest {manifest_path} is missing columns: {joined_fields}'
            )
            raise ValueError(msg)

        for line_number, row in enumerate(reader, start=2):
            kind = (row['kind'] or '').strip().lower()
            relative_path = (row['path'] or '').strip()
            if kind != 'source':
                continue
            if not relative_path:
                msg = (
                    f'Incomplete source row in {manifest_path} '
                    f'at line {line_number}'
                )
                raise ValueError(msg)

            input_path = manifest_path.parent / relative_path
            samples.append(
                InferenceSample(
                    input_path=input_path,
                    output_path=output_dir / relative_path,
                )
            )

    if not samples:
        msg = f'No source samples found in manifest: {manifest_path}'
        raise ValueError(msg)

    for sample in samples:
        if not sample.input_path.is_file():
            msg = f'Input image not found: {sample.input_path}'
            raise FileNotFoundError(msg)

    return samples
