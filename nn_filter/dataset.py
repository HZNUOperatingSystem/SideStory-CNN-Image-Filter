import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .config import ColorMode


@dataclass(frozen=True, slots=True)
class ImagePair:
    source_path: Path
    target_path: Path


class ImageRestorationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        manifest_path: Path,
        *,
        color_mode: ColorMode = 'rgb',
        patch_size: int | None = None,
        random_crop: bool = False,
    ) -> None:
        self.manifest_path = manifest_path
        self.color_mode = color_mode
        self.patch_size = patch_size
        self.random_crop = random_crop
        self.samples = self._load_samples()
        self.image_size = self._validate_samples()
        self._validate_patch_size()
        self.transform = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        with Image.open(sample.source_path) as source_image:
            source = self._convert_color_mode(source_image)
        with Image.open(sample.target_path) as target_image:
            target = self._convert_color_mode(target_image)
        source, target = self._apply_patch(source, target)
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
            msg = (
                'No image size could be determined from '
                f'{self.manifest_path}'
            )
            raise ValueError(msg)

        return reference_size

    def _validate_patch_size(self) -> None:
        if self.patch_size is None:
            return
        if self.patch_size <= 0:
            msg = f'patch_size must be positive, got {self.patch_size}'
            raise ValueError(msg)

        width, height = self.image_size
        if self.patch_size > width or self.patch_size > height:
            msg = (
                f'patch_size {self.patch_size} exceeds image size '
                f'{self.image_size} in {self.manifest_path}'
            )
            raise ValueError(msg)

    def _convert_color_mode(self, image: Image.Image) -> Image.Image:
        if self.color_mode == 'rgb':
            return image.convert('RGB')
        return image.convert('YCbCr').split()[0]

    def _apply_patch(
        self, source: Image.Image, target: Image.Image
    ) -> tuple[Image.Image, Image.Image]:
        patch_size = self._require_patch_size()
        if patch_size is None:
            return source, target

        if self.random_crop:
            top, left = self._random_crop_origin(source.size, patch_size)
        else:
            top, left = self._center_crop_origin(source.size, patch_size)

        right = left + patch_size
        bottom = top + patch_size
        crop_box = (left, top, right, bottom)
        return (
            source.crop(crop_box),
            target.crop(crop_box),
        )

    def _random_crop_origin(
        self, image_size: tuple[int, int], patch_size: int
    ) -> tuple[int, int]:
        width, height = image_size
        max_top = height - patch_size
        max_left = width - patch_size
        top = int(torch.randint(0, max_top + 1, ()).item())
        left = int(torch.randint(0, max_left + 1, ()).item())
        return top, left

    def _center_crop_origin(
        self, image_size: tuple[int, int], patch_size: int
    ) -> tuple[int, int]:
        width, height = image_size
        top = (height - patch_size) // 2
        left = (width - patch_size) // 2
        return top, left

    def _require_patch_size(self) -> int | None:
        return self.patch_size
