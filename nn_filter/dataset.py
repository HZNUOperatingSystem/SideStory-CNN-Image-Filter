import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


@dataclass(frozen=True, slots=True)
class ImagePair:
    source_path: Path
    target_path: Path


class ImageRestorationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, manifest_path: Path) -> None:
        self.manifest_path = manifest_path
        self.samples = self._load_samples()
        self.image_size = self._validate_samples()
        self.transform = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        with Image.open(sample.source_path) as source_image:
            source = source_image.convert('RGB')
        with Image.open(sample.target_path) as target_image:
            target = target_image.convert('RGB')
        return self.transform(source), self.transform(target)

    def _load_samples(self) -> list[ImagePair]:
        grouped_rows: dict[str, dict[str, Path]] = defaultdict(dict)

        with self.manifest_path.open(newline='') as manifest_file:
            reader = csv.DictReader(manifest_file)
            fieldnames = reader.fieldnames or []
            required_fields = {'sample', 'kind', 'path'}
            missing_fields = sorted(required_fields - set(fieldnames))
            if missing_fields:
                joined_fields = ', '.join(missing_fields)
                msg = (
                    f'Manifest {self.manifest_path} is missing columns: '
                    f'{joined_fields}'
                )
                raise ValueError(msg)

            for line_number, row in enumerate(reader, start=2):
                sample_name = (row['sample'] or '').strip()
                kind = (row['kind'] or '').strip().lower()
                relative_path = (row['path'] or '').strip()

                if not sample_name or not kind or not relative_path:
                    msg = (
                        f'Incomplete row in {self.manifest_path} '
                        f'at line {line_number}'
                    )
                    raise ValueError(msg)

                if kind not in {'source', 'target'}:
                    msg = (
                        f'Invalid kind {kind!r} in {self.manifest_path} '
                        f'at line {line_number}'
                    )
                    raise ValueError(msg)

                sample_rows = grouped_rows[sample_name]
                if kind in sample_rows:
                    msg = (
                        f'Duplicate {kind!r} entry for sample '
                        f'{sample_name!r} in {self.manifest_path}'
                    )
                    raise ValueError(msg)
                sample_rows[kind] = self.manifest_path.parent / relative_path

        if not grouped_rows:
            msg = f'No samples found in manifest: {self.manifest_path}'
            raise ValueError(msg)

        samples: list[ImagePair] = []
        for sample_name in sorted(grouped_rows):
            sample_rows = grouped_rows[sample_name]
            missing_kinds = sorted({'source', 'target'} - set(sample_rows))
            if missing_kinds:
                joined_kinds = ', '.join(missing_kinds)
                msg = (
                    f'Sample {sample_name!r} in {self.manifest_path} '
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

    def _validate_samples(self) -> tuple[int, int]:
        reference_size: tuple[int, int] | None = None

        for sample in self.samples:
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

            if reference_size is None:
                reference_size = source_size
                continue

            if source_size != reference_size:
                msg = (
                    'Inconsistent input image size in '
                    f'{self.manifest_path}: expected {reference_size}, '
                    f'got {source_size} for {sample.source_path}'
                )
                raise ValueError(msg)

        if reference_size is None:
            msg = f'No image size could be determined from {self.manifest_path}'
            raise ValueError(msg)

        return reference_size
