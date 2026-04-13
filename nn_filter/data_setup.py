import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .config import ColorMode
from .dataset import ImagePair, ImageRestorationDataset


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
    random_crop: bool = False,
) -> tuple[ImageRestorationDataset, DatasetSummary]:
    samples = load_image_pairs(manifest_path)
    summary = validate_image_pairs(
        samples, manifest_path=manifest_path, patch_size=patch_size
    )
    dataset = ImageRestorationDataset(
        samples,
        color_mode=color_mode,
        patch_size=patch_size,
        random_crop=random_crop,
    )
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

    samples: list[ImagePair] = []
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

        samples.append(
            ImagePair(
                source_path=sample_rows['source'],
                target_path=sample_rows['target'],
            )
        )

    return samples


def validate_image_pairs(
    samples: list[ImagePair],
    *,
    manifest_path: Path,
    patch_size: int | None = None,
) -> DatasetSummary:
    if patch_size is not None and patch_size <= 0:
        msg = f'patch_size must be positive, got {patch_size}'
        raise ValueError(msg)

    image_sizes: set[tuple[int, int]] = set()
    for sample in samples:
        if not sample.source_path.is_file():
            msg = f'Source image not found: {sample.source_path}'
            raise FileNotFoundError(msg)
        if not sample.target_path.is_file():
            msg = f'Target image not found: {sample.target_path}'
            raise FileNotFoundError(msg)

        with Image.open(sample.source_path) as source_image:
            source_size = source_image.size
        with Image.open(sample.target_path) as target_image:
            target_size = target_image.size

        if source_size != target_size:
            msg = (
                'Mismatched paired image size for '
                f'{sample.source_path} and {sample.target_path}: '
                f'{source_size} vs {target_size}'
            )
            raise ValueError(msg)

        image_sizes.add(source_size)

        if patch_size is not None:
            width, height = source_size
            if patch_size > width or patch_size > height:
                msg = (
                    f'patch_size {patch_size} exceeds image size '
                    f'{source_size} for {sample.source_path}'
                )
                raise ValueError(msg)

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
