from dataclasses import dataclass
from pathlib import Path

import torch
from rich.text import Text

from .config import ColorMode, InferConfig
from .infer_setup import (
    load_checkpoint,
    load_inference_samples,
)
from .io_utils import load_image_tensor, save_image_tensor
from .metrics import (
    gray_psnr_value,
    gray_ssim_value,
    rgb_psnr_value,
    rgb_ssim_value,
    y_psnr_value,
    y_ssim_value,
)
from .runtime import get_device
from .status import (
    ResolvedStatusConfig,
    anchor_required_metrics,
    resolve_status_config,
    resolve_status_value,
)
from .status_ui import format_named_values_line
from .ui import print_device, print_text, progress

DEVICE_METRIC_FUNCTIONS = {
    'gray_psnr': gray_psnr_value,
    'gray_ssim': gray_ssim_value,
    'rgb_psnr': rgb_psnr_value,
    'rgb_ssim': rgb_ssim_value,
    'y_psnr': y_psnr_value,
    'y_ssim': y_ssim_value,
}


@dataclass(slots=True)
class InferMetricState:
    metric_sums: dict[str, torch.Tensor]
    anchor_metric_sums: dict[str, torch.Tensor]
    sample_count: int = 0


def infer_model(
    config: InferConfig, *, device: torch.device | None = None
) -> None:
    inference_device = device if device is not None else get_device()
    loaded_checkpoint = load_checkpoint(config, device=inference_device)
    status_config = _build_infer_status_config(loaded_checkpoint.color_mode)
    samples = load_inference_samples(
        config.input,
        output_dir=loaded_checkpoint.output_dir,
    )
    metric_state = _build_infer_metric_state(
        status_config,
        device=inference_device,
    )

    print_device(inference_device)
    for sample in progress(samples, desc='infer'):
        input_cpu = load_image_tensor(
            sample.input_path,
            color_mode=loaded_checkpoint.color_mode,
        )
        target_cpu = (
            load_image_tensor(
                sample.target_path,
                color_mode=loaded_checkpoint.color_mode,
            )
            if sample.target_path is not None
            else None
        )
        input_tensor = input_cpu.unsqueeze(0).to(inference_device)
        with torch.inference_mode():
            output_tensor = loaded_checkpoint.model(input_tensor)
        output_cpu = output_tensor.squeeze(0).detach().cpu()
        sample.output_path.parent.mkdir(parents=True, exist_ok=True)
        save_image_tensor(output_cpu, sample.output_path)
        if target_cpu is not None:
            target_tensor = target_cpu.unsqueeze(0).to(inference_device)
            _update_infer_metric_state(
                metric_state,
                prediction=output_tensor,
                target=target_tensor,
                anchor=input_tensor,
            )

    print_text(
        _build_infer_summary(
            checkpoint_path=loaded_checkpoint.checkpoint_path,
            output_dir=loaded_checkpoint.output_dir,
            sample_count=len(samples),
        )
    )
    metric_summary = _finish_infer_metric_state(
        metric_state,
        status_config=status_config,
    )
    if metric_summary:
        print_text(
            format_named_values_line(
                'metrics',
                metric_summary,
                selected_statuses=status_config.selected_statuses,
            )
        )


def _build_infer_status_config(color_mode: ColorMode) -> ResolvedStatusConfig:
    return resolve_status_config(
        _infer_status_names(color_mode),
        target_value=None,
        watched_best=[],
        color_mode=color_mode,
    )


def _infer_status_names(color_mode: ColorMode) -> list[str]:
    if color_mode == 'rgb':
        return [
            'y_psnr',
            'y_psnr_gain',
            'rgb_psnr',
            'rgb_psnr_gain',
            'y_ssim',
            'y_ssim_gain',
            'rgb_ssim',
            'rgb_ssim_gain',
        ]
    return [
        'y_psnr',
        'y_psnr_gain',
        'gray_psnr',
        'gray_psnr_gain',
        'y_ssim',
        'y_ssim_gain',
        'gray_ssim',
        'gray_ssim_gain',
    ]


def _build_infer_metric_state(
    status_config: ResolvedStatusConfig,
    *,
    device: torch.device,
) -> InferMetricState:
    metric_sums = {
        name: torch.zeros((), device=device, dtype=torch.float32)
        for name in status_config.selected_metrics
    }
    anchor_metric_names = anchor_required_metrics(
        status_config.selected_statuses
    )
    anchor_metric_sums = {
        name: torch.zeros((), device=device, dtype=torch.float32)
        for name in anchor_metric_names
    }
    return InferMetricState(
        metric_sums=metric_sums,
        anchor_metric_sums=anchor_metric_sums,
    )


def _update_infer_metric_state(
    metric_state: InferMetricState,
    *,
    prediction: torch.Tensor,
    target: torch.Tensor,
    anchor: torch.Tensor,
) -> None:
    for metric_name, metric_sum in metric_state.metric_sums.items():
        metric_fn = DEVICE_METRIC_FUNCTIONS[metric_name]
        metric_value = metric_fn(prediction, target).to(metric_sum.dtype)
        metric_state.metric_sums[metric_name] = metric_sum + metric_value

    for metric_name, metric_sum in metric_state.anchor_metric_sums.items():
        metric_fn = DEVICE_METRIC_FUNCTIONS[metric_name]
        metric_value = metric_fn(anchor, target).to(metric_sum.dtype)
        metric_state.anchor_metric_sums[metric_name] = metric_sum + metric_value

    metric_state.sample_count += 1


def _finish_infer_metric_state(
    metric_state: InferMetricState,
    *,
    status_config: ResolvedStatusConfig,
) -> dict[str, float]:
    if metric_state.sample_count == 0:
        return {}

    current_metrics = {
        name: (metric_sum / metric_state.sample_count).item()
        for name, metric_sum in metric_state.metric_sums.items()
    }
    anchor_metrics = {
        name: (metric_sum / metric_state.sample_count).item()
        for name, metric_sum in metric_state.anchor_metric_sums.items()
    }
    return {
        status_name: resolve_status_value(
            status_name,
            current_metrics=current_metrics,
            anchor_metrics=anchor_metrics,
        )
        for status_name in status_config.selected_statuses
    }


def _build_infer_summary(
    *,
    checkpoint_path: Path,
    output_dir: Path,
    sample_count: int,
) -> Text:
    summary = Text()
    summary.append('infer', style='bold blue')
    summary.append(': ')
    summary.append(str(checkpoint_path), style='green')
    summary.append(' | ', style='dim')
    summary.append(str(output_dir), style='yellow')
    summary.append(' | ', style='dim')
    summary.append(f'samples={sample_count}', style='magenta')
    return summary
