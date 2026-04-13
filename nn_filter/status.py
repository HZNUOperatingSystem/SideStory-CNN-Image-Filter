from __future__ import annotations

import math
from collections.abc import Callable, Mapping

import torch
from rich.text import Text

from .config import ColorMode, StatusSelection
from .metrics import batch_vmaf, gray_psnr, rgb_psnr, y_psnr

MetricFunction = Callable[[torch.Tensor, torch.Tensor], float]

BASE_METRIC_FUNCTIONS: dict[str, MetricFunction] = {
    'gray_psnr': gray_psnr,
    'rgb_psnr': rgb_psnr,
    'vmaf': batch_vmaf,
    'y_psnr': y_psnr,
}

COLOR_BASE_METRICS: dict[ColorMode, list[str]] = {
    'rgb': ['y_psnr', 'rgb_psnr', 'vmaf'],
    'y-only': ['y_psnr', 'gray_psnr', 'vmaf'],
}


class StatusTracker:
    def __init__(
        self,
        selection: StatusSelection,
        *,
        color_mode: ColorMode,
    ) -> None:
        self.selected_statuses = resolve_status_selection(
            selection, color_mode=color_mode
        )
        self.required_metrics = required_metrics(self.selected_statuses)
        self.metric_sums = {
            metric_name: 0.0 for metric_name in self.required_metrics
        }
        self.sample_count = 0
        self.anchor_metrics: dict[str, float] = {}
        self.best_metrics: dict[str, float] = {}

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        batch_size: int,
    ) -> None:
        if not self.selected_statuses:
            return

        for metric_name in self.required_metrics:
            metric_fn = BASE_METRIC_FUNCTIONS[metric_name]
            metric_value = metric_fn(prediction, target)
            self.metric_sums[metric_name] += metric_value * batch_size
        self.sample_count += batch_size

    def finish_epoch(self) -> Text | None:
        if not self.selected_statuses or self.sample_count == 0:
            return None

        current_metrics = {
            metric_name: total / self.sample_count
            for metric_name, total in self.metric_sums.items()
        }
        for metric_name, metric_value in current_metrics.items():
            self.anchor_metrics.setdefault(metric_name, metric_value)
            self.best_metrics[metric_name] = max(
                self.best_metrics.get(metric_name, metric_value),
                metric_value,
            )

        status_values = {
            status_name: self._resolve_status_value(
                status_name, current_metrics
            )
            for status_name in self.selected_statuses
        }
        self._reset_epoch()
        return format_status_line(status_values)

    def _resolve_status_value(
        self,
        status_name: str,
        current_metrics: Mapping[str, float],
    ) -> float:
        if status_name in current_metrics:
            return current_metrics[status_name]
        if status_name.startswith('best_'):
            return self.best_metrics[status_name.removeprefix('best_')]
        if status_name.startswith('anchor_'):
            return self.anchor_metrics[status_name.removeprefix('anchor_')]
        if status_name.endswith('_gain'):
            metric_name = status_name.removesuffix('_gain')
            return (
                current_metrics[metric_name] - self.anchor_metrics[metric_name]
            )
        msg = f'Unknown status name: {status_name}'
        raise ValueError(msg)

    def _reset_epoch(self) -> None:
        self.metric_sums = {
            metric_name: 0.0 for metric_name in self.required_metrics
        }
        self.sample_count = 0


def resolve_status_selection(
    selection: StatusSelection,
    *,
    color_mode: ColorMode,
) -> list[str]:
    compatible_metrics = COLOR_BASE_METRICS[color_mode]
    all_statuses = expand_status_names(compatible_metrics)

    if selection is False:
        return []
    if selection is True:
        return all_statuses

    invalid_statuses = sorted(set(selection) - set(all_statuses))
    if invalid_statuses:
        joined = ', '.join(invalid_statuses)
        msg = f'Unsupported statuses for color mode {color_mode}: {joined}'
        raise ValueError(msg)
    return selection


def expand_status_names(base_metrics: list[str]) -> list[str]:
    expanded: list[str] = []
    for metric_name in base_metrics:
        expanded.extend(
            [
                metric_name,
                f'best_{metric_name}',
                f'anchor_{metric_name}',
                f'{metric_name}_gain',
            ]
        )
    return expanded


def required_metrics(selected_statuses: list[str]) -> list[str]:
    metrics: list[str] = []
    for status_name in selected_statuses:
        metric_name = base_metric_name(status_name)
        if metric_name not in metrics:
            metrics.append(metric_name)
    return metrics


def base_metric_name(status_name: str) -> str:
    if status_name.startswith('best_'):
        return status_name.removeprefix('best_')
    if status_name.startswith('anchor_'):
        return status_name.removeprefix('anchor_')
    if status_name.endswith('_gain'):
        return status_name.removesuffix('_gain')
    return status_name


def format_status_line(
    status_values: Mapping[str, float],
) -> Text:
    line = Text()
    line.append('status', style='bold magenta')
    line.append(': ')
    first = True
    for name, value in status_values.items():
        if not first:
            line.append(' | ')
        first = False
        line.append(name, style='bold cyan')
        line.append('=')
        line.append_text(format_status_value(name, value))
    return line


def format_status_value(status_name: str, value: float) -> Text:
    if math.isinf(value):
        text = 'inf'
    elif status_name.endswith('_gain'):
        text = f'{value:+.4f}'
    else:
        text = f'{value:.4f}'

    rendered = Text(text)
    if status_name.endswith('_gain'):
        if value > 0:
            rendered.stylize('bold green')
            return rendered
        if value < 0:
            rendered.stylize('bold red')
            return rendered
    rendered.stylize('white')
    return rendered
