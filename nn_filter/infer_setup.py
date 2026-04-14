import csv
from dataclasses import dataclass
from pathlib import Path

import torch

from .checkpoint import load_model_checkpoint, resolve_run_checkpoint_path
from .config import ColorMode, InferConfig
from .io_utils import is_image_path


@dataclass(frozen=True, slots=True)
class InferenceSample:
    input_path: Path
    output_path: Path
    target_path: Path | None = None


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
def _load_manifest_samples(
    manifest_path: Path,
    *,
    output_dir: Path,
) -> list[InferenceSample]:
    samples: list[InferenceSample] = []
    grouped_rows: dict[str, dict[str, Path]] = {}
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
            sample_name = (row['sample'] or '').strip()
            kind = (row['kind'] or '').strip().lower()
            relative_path = (row['path'] or '').strip()
            if not sample_name or not kind or not relative_path:
                msg = f'Incomplete row in {manifest_path} at line {line_number}'
                raise ValueError(msg)
            if kind not in {'source', 'target'}:
                msg = (
                    f'Invalid kind {kind!r} in {manifest_path} '
                    f'at line {line_number}'
                )
                raise ValueError(msg)

            sample_rows = grouped_rows.setdefault(sample_name, {})
            if kind in sample_rows:
                msg = (
                    f'Duplicate {kind!r} entry for sample '
                    f'{sample_name!r} in {manifest_path}'
                )
                raise ValueError(msg)
            sample_rows[kind] = manifest_path.parent / relative_path

    for sample_name in sorted(grouped_rows):
        sample_rows = grouped_rows[sample_name]
        input_path = sample_rows.get('source')
        if input_path is None:
            msg = (
                f'Sample {sample_name!r} in {manifest_path} is missing: source'
            )
            raise ValueError(msg)
        target_path = sample_rows.get('target')
        output_relative_path = input_path.relative_to(manifest_path.parent)
        samples.append(
            InferenceSample(
                input_path=input_path,
                output_path=output_dir / output_relative_path,
                target_path=target_path,
            )
        )
    if not samples:
        msg = f'No source samples found in manifest: {manifest_path}'
        raise ValueError(msg)

    for sample in samples:
        if not sample.input_path.is_file():
            msg = f'Input image not found: {sample.input_path}'
            raise FileNotFoundError(msg)
        if sample.target_path is not None and not sample.target_path.is_file():
            msg = f'Target image not found: {sample.target_path}'
            raise FileNotFoundError(msg)

    return samples
