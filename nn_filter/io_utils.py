from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

EXPECTED_IMAGE_NDIM = 3
GRAY_CHANNELS = 1
RGB_CHANNELS = 3
IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
_TO_TENSOR = transforms.ToTensor()


def save_image_tensor(tensor: torch.Tensor, path: Path) -> None:
    image_tensor = tensor.detach().cpu()
    if image_tensor.ndim != EXPECTED_IMAGE_NDIM:
        msg = (
            'Expected image tensor shape [C, H, W], '
            f'got {tuple(image_tensor.shape)}'
        )
        raise ValueError(msg)

    image_tensor = image_tensor.clamp(0.0, 1.0)
    image_array = (
        image_tensor.mul(255.0).round().to(torch.uint8).permute(1, 2, 0).numpy()
    )

    if image_tensor.shape[0] == GRAY_CHANNELS:
        Image.fromarray(image_array[:, :, 0], mode='L').save(path)
        return
    if image_tensor.shape[0] == RGB_CHANNELS:
        Image.fromarray(image_array, mode='RGB').save(path)
        return

    msg = (
        'Expected image tensor with 1 or 3 channels, '
        f'got {image_tensor.shape[0]}'
    )
    raise ValueError(msg)


def load_image_tensor(
    path: Path,
    *,
    color_mode: str,
) -> torch.Tensor:
    with Image.open(path) as image:
        if color_mode == 'rgb':
            converted = image.convert('RGB')
        else:
            converted = image.convert('YCbCr').split()[0]
        return _TO_TENSOR(converted)


def is_image_path(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES
