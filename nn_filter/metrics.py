from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import torch

from .io_utils import save_image_tensor

BATCH_NDIM = 4
GRAY_CHANNELS = 1
RGB_CHANNELS = 3


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
    mse = torch.mean((prediction - target) ** 2).item()
    if mse == 0.0:
        return float('inf')
    ratio = torch.tensor((max_value * max_value) / mse)
    return 10.0 * torch.log10(ratio).item()


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


def y_psnr(prediction: torch.Tensor, target: torch.Tensor) -> float:
    prediction = _ensure_4d(prediction)
    target = _ensure_4d(target)

    if (
        prediction.shape[1] == GRAY_CHANNELS
        and target.shape[1] == GRAY_CHANNELS
    ):
        return gray_psnr(prediction, target)
    if prediction.shape[1] == RGB_CHANNELS and target.shape[1] == RGB_CHANNELS:
        return _psnr_from_tensors(
            rgb_to_limited_y601(prediction),
            rgb_to_limited_y601(target),
            max_value=255.0,
        )
    msg = (
        'y_psnr expects either 1-channel Y tensors or 3-channel RGB tensors '
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
