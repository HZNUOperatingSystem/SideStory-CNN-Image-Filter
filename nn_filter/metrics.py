from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import torch
from torch.nn import functional as F

from .io_utils import save_image_tensor

BATCH_NDIM = 4
GRAY_CHANNELS = 1
RGB_CHANNELS = 3
DEFAULT_SSIM_KERNEL_SIZE = 11
DEFAULT_SSIM_SIGMA = 1.5
SSIM_K1 = 0.01
SSIM_K2 = 0.03


def _ensure_4d(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != BATCH_NDIM:
        msg = f'Expected tensor shape [N, C, H, W], got {tuple(tensor.shape)}'
        raise ValueError(msg)
    return tensor


def _psnr_from_tensors(
    prediction: torch.Tensor,
    target: torch.Tensor,
    max_value: float = 255.0,
) -> float:
    return _psnr_value_from_tensors(
        prediction,
        target,
        max_value=max_value,
    ).item()


def _psnr_value_from_tensors(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    max_value: float = 255.0,
) -> torch.Tensor:
    mse_tensor = torch.mean((prediction - target) ** 2)
    if mse_tensor.item() == 0.0:
        return torch.tensor(float('inf'), device=prediction.device)
    ratio = (max_value * max_value) / mse_tensor
    return 10.0 * torch.log10(ratio)


def _ssim_from_tensors(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    max_value: float = 1.0,
) -> float:
    return _ssim_value_from_tensors(
        prediction,
        target,
        max_value=max_value,
    ).item()


def _ssim_value_from_tensors(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    max_value: float = 1.0,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        msg = (
            f'Shape mismatch: prediction {tuple(prediction.shape)} vs '
            f'target {tuple(target.shape)}'
        )
        raise ValueError(msg)

    kernel_size = _resolve_ssim_kernel_size(prediction)
    sigma = _resolve_ssim_sigma(kernel_size)
    channels = prediction.shape[1]
    window = _gaussian_window(
        kernel_size=kernel_size,
        sigma=sigma,
        channels=channels,
        device=prediction.device,
        dtype=prediction.dtype,
    )
    mu_prediction = _apply_window(prediction, window)
    mu_target = _apply_window(target, window)

    mu_prediction_sq = mu_prediction.square()
    mu_target_sq = mu_target.square()
    mu_prediction_target = mu_prediction * mu_target

    sigma_prediction_sq = (
        _apply_window(prediction.square(), window) - mu_prediction_sq
    )
    sigma_target_sq = _apply_window(target.square(), window) - mu_target_sq
    sigma_prediction_target = (
        _apply_window(prediction * target, window) - mu_prediction_target
    )

    c1 = (SSIM_K1 * max_value) ** 2
    c2 = (SSIM_K2 * max_value) ** 2
    numerator = (2.0 * mu_prediction_target + c1) * (
        2.0 * sigma_prediction_target + c2
    )
    denominator = (mu_prediction_sq + mu_target_sq + c1) * (
        sigma_prediction_sq + sigma_target_sq + c2
    )
    ssim_map = numerator / denominator.clamp_min(torch.finfo(window.dtype).eps)
    return ssim_map.mean()


def _resolve_ssim_kernel_size(tensor: torch.Tensor) -> int:
    spatial_limit = min(
        DEFAULT_SSIM_KERNEL_SIZE,
        tensor.shape[2],
        tensor.shape[3],
    )
    if spatial_limit % 2 == 0:
        spatial_limit -= 1
    return max(1, spatial_limit)


def _resolve_ssim_sigma(kernel_size: int) -> float:
    if kernel_size == 1:
        return 1.0
    return DEFAULT_SSIM_SIGMA * kernel_size / DEFAULT_SSIM_KERNEL_SIZE


def _gaussian_window(
    *,
    kernel_size: int,
    sigma: float,
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    positions = torch.arange(kernel_size, device=device, dtype=dtype)
    centered = positions - (kernel_size - 1) / 2.0
    kernel_1d = torch.exp(-(centered.square()) / (2.0 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d.expand(channels, 1, kernel_size, kernel_size).contiguous()


def _apply_window(tensor: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    padding = window.shape[-1] // 2
    if padding == 0:
        return F.conv2d(tensor, window, groups=tensor.shape[1])
    padded = F.pad(
        tensor,
        (padding, padding, padding, padding),
        mode='replicate',
    )
    return F.conv2d(padded, window, groups=tensor.shape[1])


def rgb_to_limited_y601(tensor: torch.Tensor) -> torch.Tensor:
    tensor = _ensure_4d(tensor)
    if tensor.shape[1] != RGB_CHANNELS:
        msg = f'Expected RGB tensor with 3 channels, got {tuple(tensor.shape)}'
        raise ValueError(msg)

    r = tensor[:, 0:1, :, :]
    g = tensor[:, 1:2, :, :]
    b = tensor[:, 2:3, :, :]
    return 16.0 + 65.481 * r + 128.553 * g + 24.966 * b


def rgb_psnr(prediction: torch.Tensor, target: torch.Tensor) -> float:
    prediction = _ensure_4d(prediction)
    target = _ensure_4d(target)
    if prediction.shape[1] != RGB_CHANNELS or target.shape[1] != RGB_CHANNELS:
        raise ValueError('rgb_psnr expects 3-channel RGB tensors')
    return _psnr_from_tensors(
        prediction * 255.0,
        target * 255.0,
        max_value=255.0,
    )


def rgb_psnr_value(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    prediction = _ensure_4d(prediction)
    target = _ensure_4d(target)
    if prediction.shape[1] != RGB_CHANNELS or target.shape[1] != RGB_CHANNELS:
        raise ValueError('rgb_psnr expects 3-channel RGB tensors')
    return _psnr_value_from_tensors(
        prediction * 255.0,
        target * 255.0,
        max_value=255.0,
    )


def rgb_ssim(prediction: torch.Tensor, target: torch.Tensor) -> float:
    prediction = _ensure_4d(prediction)
    target = _ensure_4d(target)
    if prediction.shape[1] != RGB_CHANNELS or target.shape[1] != RGB_CHANNELS:
        raise ValueError('rgb_ssim expects 3-channel RGB tensors')
    return _ssim_from_tensors(prediction, target, max_value=1.0)


def rgb_ssim_value(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    prediction = _ensure_4d(prediction)
    target = _ensure_4d(target)
    if prediction.shape[1] != RGB_CHANNELS or target.shape[1] != RGB_CHANNELS:
        raise ValueError('rgb_ssim expects 3-channel RGB tensors')
    return _ssim_value_from_tensors(prediction, target, max_value=1.0)


def gray_psnr(prediction: torch.Tensor, target: torch.Tensor) -> float:
    prediction = _ensure_4d(prediction)
    target = _ensure_4d(target)
    if prediction.shape[1] != GRAY_CHANNELS or target.shape[1] != GRAY_CHANNELS:
        raise ValueError('gray_psnr expects 1-channel tensors')
    return _psnr_from_tensors(
        prediction * 255.0,
        target * 255.0,
        max_value=255.0,
    )


def gray_psnr_value(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    prediction = _ensure_4d(prediction)
    target = _ensure_4d(target)
    if prediction.shape[1] != GRAY_CHANNELS or target.shape[1] != GRAY_CHANNELS:
        raise ValueError('gray_psnr expects 1-channel tensors')
    return _psnr_value_from_tensors(
        prediction * 255.0,
        target * 255.0,
        max_value=255.0,
    )


def gray_ssim(prediction: torch.Tensor, target: torch.Tensor) -> float:
    prediction = _ensure_4d(prediction)
    target = _ensure_4d(target)
    if prediction.shape[1] != GRAY_CHANNELS or target.shape[1] != GRAY_CHANNELS:
        raise ValueError('gray_ssim expects 1-channel tensors')
    return _ssim_from_tensors(prediction, target, max_value=1.0)


def gray_ssim_value(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    prediction = _ensure_4d(prediction)
    target = _ensure_4d(target)
    if prediction.shape[1] != GRAY_CHANNELS or target.shape[1] != GRAY_CHANNELS:
        raise ValueError('gray_ssim expects 1-channel tensors')
    return _ssim_value_from_tensors(prediction, target, max_value=1.0)


def y_psnr(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return y_psnr_value(prediction, target).item()


def y_psnr_value(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    prediction = _ensure_4d(prediction)
    target = _ensure_4d(target)

    if (
        prediction.shape[1] == GRAY_CHANNELS
        and target.shape[1] == GRAY_CHANNELS
    ):
        return gray_psnr_value(prediction, target)
    if prediction.shape[1] == RGB_CHANNELS and target.shape[1] == RGB_CHANNELS:
        return _psnr_value_from_tensors(
            rgb_to_limited_y601(prediction),
            rgb_to_limited_y601(target),
            max_value=255.0,
        )
    msg = (
        'y_psnr expects either 1-channel Y tensors or 3-channel RGB tensors '
        'with matching shapes'
    )
    raise ValueError(msg)


def y_ssim(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return y_ssim_value(prediction, target).item()


def y_ssim_value(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    prediction = _ensure_4d(prediction)
    target = _ensure_4d(target)

    if (
        prediction.shape[1] == GRAY_CHANNELS
        and target.shape[1] == GRAY_CHANNELS
    ):
        return gray_ssim_value(prediction, target)
    if prediction.shape[1] == RGB_CHANNELS and target.shape[1] == RGB_CHANNELS:
        return _ssim_value_from_tensors(
            rgb_to_limited_y601(prediction) / 255.0,
            rgb_to_limited_y601(target) / 255.0,
            max_value=1.0,
        )
    msg = (
        'y_ssim expects either 1-channel Y tensors or 3-channel RGB tensors '
        'with matching shapes'
    )
    raise ValueError(msg)


def calculate_vmaf(
    distorted: torch.Tensor,
    reference: torch.Tensor,
    vmaf_model: str = 'vmaf_v0.6.1',
) -> float:
    distorted = distorted.detach().cpu()
    reference = reference.detach().cpu()

    if distorted.ndim == BATCH_NDIM:
        if distorted.shape[0] != 1:
            msg = 'calculate_vmaf expects a single image when given 4D tensors'
            raise ValueError(msg)
        distorted = distorted.squeeze(0)
    if reference.ndim == BATCH_NDIM:
        if reference.shape[0] != 1:
            msg = 'calculate_vmaf expects a single image when given 4D tensors'
            raise ValueError(msg)
        reference = reference.squeeze(0)

    if distorted.shape != reference.shape:
        msg = (
            f'Shape mismatch: distorted {tuple(distorted.shape)} vs '
            f'reference {tuple(reference.shape)}'
        )
        raise ValueError(msg)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        distorted_path = tmp_path / 'distorted.png'
        reference_path = tmp_path / 'reference.png'

        save_image_tensor(distorted, distorted_path)
        save_image_tensor(reference, reference_path)

        cmd = [
            'ffmpeg',
            '-hide_banner',
            '-loglevel',
            'error',
            '-i',
            str(distorted_path),
            '-i',
            str(reference_path),
            '-lavfi',
            f'libvmaf=model={vmaf_model}:log_path=/dev/null',
            '-f',
            'null',
            '-',
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as error:
            msg = (
                'VMAF calculation failed. Ensure FFmpeg is built with libvmaf. '
                f'Error: {error.stderr}'
            )
            raise RuntimeError(msg) from error
        except FileNotFoundError as error:
            msg = (
                'FFmpeg not found. Please install FFmpeg with libvmaf support.'
            )
            raise RuntimeError(msg) from error

        for line in result.stderr.split('\n'):
            if 'VMAF score:' in line:
                score_str = line.split('VMAF score:')[-1].strip()
                try:
                    return float(score_str)
                except ValueError:
                    continue

        msg = f'Could not parse VMAF score from FFmpeg output: {result.stderr}'
        raise RuntimeError(msg)


def batch_vmaf(prediction: torch.Tensor, target: torch.Tensor) -> float:
    prediction = _ensure_4d(prediction)
    target = _ensure_4d(target)
    if prediction.shape != target.shape:
        msg = (
            f'Shape mismatch: prediction {tuple(prediction.shape)} vs '
            f'target {tuple(target.shape)}'
        )
        raise ValueError(msg)

    scores = [
        calculate_vmaf(
            prediction[index : index + 1],
            target[index : index + 1],
        )
        for index in range(prediction.shape[0])
    ]
    return sum(scores) / len(scores)


def ensure_vmaf_runtime_available() -> None:
    cmd = [
        'ffmpeg',
        '-hide_banner',
        '-filters',
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as error:
        msg = f'Failed to inspect FFmpeg filters: {error.stderr}'
        raise RuntimeError(msg) from error
    except FileNotFoundError as error:
        msg = 'FFmpeg not found. Please install FFmpeg with libvmaf support.'
        raise RuntimeError(msg) from error

    filters_output = f'{result.stdout}\n{result.stderr}'
    if 'libvmaf' not in filters_output:
        msg = 'FFmpeg is available, but the libvmaf filter is missing.'
        raise RuntimeError(msg)
