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
        samples: list[ImagePair],
        *,
        color_mode: ColorMode = 'rgb',
        patch_size: int | None = None,
        random_crop: bool = False,
    ) -> None:
        self.samples = samples
        self.color_mode = color_mode
        self.patch_size = patch_size
        self.random_crop = random_crop
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
