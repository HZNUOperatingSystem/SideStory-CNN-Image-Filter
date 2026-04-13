import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .config import ColorMode
from .dataset import (
    CachedImagePair,
    DatasetSample,
    ImagePair,
    ImageRestorationDataset,
)
from .io_utils import load_image_tensor

DatasetBuildMode = Literal['full-image', 'patch-grid']


@dataclass(frozen=True, slots=True)
class DatasetSummary:
    image_size: tuple[int, int] | None
    resolution_count: int

    @property
    def has_mixed_resolution(self) -> bool:
        return self.image_size is None

    @property
    def image_size_label(self) -> str:
        if self.image_size is not None:
            return str(self.image_size)
        return f'mixed({self.resolution_count})'


def create_dataset(
    manifest_path: Path,
    *,
    color_mode: ColorMode = 'rgb',
    patch_size: int | None = None,
    build_mode: DatasetBuildMode = 'full-image',
) -> tuple[ImageRestorationDataset, DatasetSummary]:
    pairs = load_image_pairs(manifest_path)
    cached_pairs, summary = cache_image_pairs(
        pairs,
        manifest_path=manifest_path,
        color_mode=color_mode,
        patch_size=patch_size if build_mode == 'patch-grid' else None,
    )
    samples = build_dataset_samples(
        cached_pairs,
        manifest_path=manifest_path,
        build_mode=build_mode,
        patch_size=patch_size,
    )
    dataset = ImageRestorationDataset(cached_pairs, samples)
    return dataset, summary


def load_image_pairs(manifest_path: Path) -> list[ImagePair]:
    grouped_rows: dict[str, dict[str, Path]] = defaultdict(dict)

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

            sample_rows = grouped_rows[sample_name]
            if kind in sample_rows:
                msg = (
                    f'Duplicate {kind!r} entry for sample '
                    f'{sample_name!r} in {manifest_path}'
                )
                raise ValueError(msg)
            sample_rows[kind] = manifest_path.parent / relative_path

    if not grouped_rows:
        msg = f'No samples found in manifest: {manifest_path}'
        raise ValueError(msg)

    pairs: list[ImagePair] = []
    for sample_name in sorted(grouped_rows):
        sample_rows = grouped_rows[sample_name]
        missing_kinds = sorted({'source', 'target'} - set(sample_rows))
        if missing_kinds:
            joined_kinds = ', '.join(missing_kinds)
            msg = (
                f'Sample {sample_name!r} in {manifest_path} '
                f'is missing: {joined_kinds}'
            )
            raise ValueError(msg)

        pairs.append(
            ImagePair(
                source_path=sample_rows['source'],
                target_path=sample_rows['target'],
            )
        )

    return pairs


def cache_image_pairs(
    pairs: list[ImagePair],
    *,
    manifest_path: Path,
    color_mode: ColorMode,
    patch_size: int | None = None,
) -> tuple[list[CachedImagePair], DatasetSummary]:
    if patch_size is not None and patch_size <= 0:
        msg = f'patch_size must be positive, got {patch_size}'
        raise ValueError(msg)

    image_sizes: set[tuple[int, int]] = set()
    cached_pairs: list[CachedImagePair] = []
    for pair in pairs:
        if not pair.source_path.is_file():
            msg = f'Source image not found: {pair.source_path}'
            raise FileNotFoundError(msg)
        if not pair.target_path.is_file():
            msg = f'Target image not found: {pair.target_path}'
            raise FileNotFoundError(msg)

        source = load_image_tensor(pair.source_path, color_mode=color_mode)
        target = load_image_tensor(pair.target_path, color_mode=color_mode)
        if source.shape != target.shape:
            msg = (
                'Mismatched paired image size for '
                f'{pair.source_path} and {pair.target_path}: '
                f'{tuple(source.shape)} vs {tuple(target.shape)}'
            )
            raise ValueError(msg)

        height, width = source.shape[1:]
        image_sizes.add((width, height))
        if patch_size is not None and (
            patch_size > width or patch_size > height
        ):
            msg = (
                f'patch_size {patch_size} exceeds image size '
                f'{(width, height)} for {pair.source_path}'
            )
            raise ValueError(msg)

        cached_pairs.append(
            CachedImagePair(
                source=source.contiguous(),
                target=target.contiguous(),
            )
        )

    summary = build_dataset_summary(image_sizes, manifest_path=manifest_path)
    return cached_pairs, summary


def build_dataset_samples(
    pairs: list[CachedImagePair],
    *,
    manifest_path: Path,
    build_mode: DatasetBuildMode,
    patch_size: int | None,
) -> list[DatasetSample]:
    if build_mode == 'full-image':
        return [DatasetSample(pair_index=index) for index in range(len(pairs))]

    if patch_size is None or patch_size <= 0:
        msg = 'patch_size must be a positive integer for patch-grid mode'
        raise ValueError(msg)

    samples: list[DatasetSample] = []
    for pair_index, pair in enumerate(pairs):
        _, height, width = pair.source.shape
        for top in range(0, height - patch_size + 1, patch_size):
            for left in range(0, width - patch_size + 1, patch_size):
                samples.append(
                    DatasetSample(
                        pair_index=pair_index,
                        top=top,
                        left=left,
                        size=patch_size,
                    )
                )

    if not samples:
        msg = f'No dataset samples could be built from {manifest_path}'
        raise ValueError(msg)
    return samples


def build_dataset_summary(
    image_sizes: set[tuple[int, int]],
    *,
    manifest_path: Path,
) -> DatasetSummary:
    if not image_sizes:
        msg = f'No image size could be determined from {manifest_path}'
        raise ValueError(msg)

    if len(image_sizes) == 1:
        return DatasetSummary(
            image_size=next(iter(image_sizes)),
            resolution_count=1,
        )

    return DatasetSummary(
        image_size=None,
        resolution_count=len(image_sizes),
    )
