from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch
from rich.text import Text

from .config import ColorMode, StatusSelection
from .metrics import (
    batch_vmaf,
    ensure_vmaf_runtime_available,
    gray_psnr,
    rgb_psnr,
    y_psnr,
)

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


@dataclass(slots=True)
class ResolvedStatusConfig:
    selected_statuses: list[str]
    selected_metrics: list[str]
    target_value: str | None
    watched_best_statuses: list[str]


@dataclass(slots=True)
class StatusSummary:
    current_metrics: dict[str, float]
    best_metrics: dict[str, float]
    status_values: dict[str, float]
    best_status_values: dict[str, float]


class StatusTracker:
    def __init__(self, status_config: ResolvedStatusConfig) -> None:
        self.status_config = status_config
        self.required_metrics = list(status_config.selected_metrics)
        self.anchor_required_metrics = anchor_required_metrics(
            status_config.selected_statuses
        )
        self.metric_sums = {
            metric_name: 0.0 for metric_name in self.required_metrics
        }
        self.anchor_metric_sums = {
            metric_name: 0.0 for metric_name in self.anchor_required_metrics
        }
        self.sample_count = 0
        self.best_metrics: dict[str, float] = {}
        self.best_status_values: dict[str, float] = {}

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        anchor: torch.Tensor | None = None,
        batch_size: int,
    ) -> None:
        if not self.status_config.selected_metrics:
            return

        for metric_name in self.required_metrics:
            metric_fn = BASE_METRIC_FUNCTIONS[metric_name]
            metric_value = metric_fn(prediction, target)
            self.metric_sums[metric_name] += metric_value * batch_size
        if self.anchor_required_metrics:
            if anchor is None:
                msg = 'anchor tensor is required for selected status values'
                raise ValueError(msg)
            for metric_name in self.anchor_required_metrics:
                metric_fn = BASE_METRIC_FUNCTIONS[metric_name]
                metric_value = metric_fn(anchor, target)
                self.anchor_metric_sums[metric_name] += (
                    metric_value * batch_size
                )
        self.sample_count += batch_size

    def finish_epoch(self) -> StatusSummary:
        if not self.status_config.selected_metrics or self.sample_count == 0:
            return StatusSummary(
                current_metrics={},
                best_metrics={},
                status_values={},
                best_status_values={},
            )

        current_metrics = {
            metric_name: total / self.sample_count
            for metric_name, total in self.metric_sums.items()
        }
        anchor_metrics = {
            metric_name: total / self.sample_count
            for metric_name, total in self.anchor_metric_sums.items()
        }
        for metric_name, metric_value in current_metrics.items():
            self.best_metrics[metric_name] = max(
                self.best_metrics.get(metric_name, metric_value),
                metric_value,
            )

        status_values = {
            status_name: resolve_status_value(
                status_name,
                current_metrics=current_metrics,
                anchor_metrics=anchor_metrics,
            )
            for status_name in self.status_config.selected_statuses
        }
        for status_name, status_value in status_values.items():
            self.best_status_values[status_name] = max(
                self.best_status_values.get(status_name, status_value),
                status_value,
            )
        self._reset_epoch()
        return StatusSummary(
            current_metrics=current_metrics,
            best_metrics=dict(self.best_metrics),
            status_values=status_values,
            best_status_values=dict(self.best_status_values),
        )

    def _reset_epoch(self) -> None:
        self.metric_sums = {
            metric_name: 0.0 for metric_name in self.required_metrics
        }
        self.anchor_metric_sums = {
            metric_name: 0.0 for metric_name in self.anchor_required_metrics
        }
        self.sample_count = 0


def resolve_status_config(
    selection: StatusSelection,
    *,
    target_value: str | None,
    watched_best: list[str],
    color_mode: ColorMode,
) -> ResolvedStatusConfig:
    compatible_metrics = COLOR_BASE_METRICS[color_mode]
    compatible_statuses = expand_status_names(compatible_metrics)

    if selection is False:
        selected_statuses: list[str] = []
        selected_metrics: list[str] = []
    elif selection is True:
        selected_statuses = list(compatible_statuses)
        selected_metrics = list(compatible_metrics)
    else:
        invalid_statuses = sorted(set(selection) - set(compatible_statuses))
        if invalid_statuses:
            joined = ', '.join(invalid_statuses)
            msg = f'Unsupported statuses for color mode {color_mode}: {joined}'
            raise ValueError(msg)
        selected_statuses = list(selection)
        selected_metrics = required_metrics(selected_statuses)

    if target_value is not None and target_value not in selected_metrics:
        msg = f'target_value must also be enabled in status: {target_value}'
        raise ValueError(msg)

    invalid_best = sorted(set(watched_best) - set(selected_statuses))
    if invalid_best:
        joined = ', '.join(invalid_best)
        msg = f'watched_best statuses must also be enabled in status: {joined}'
        raise ValueError(msg)

    return ResolvedStatusConfig(
        selected_statuses=selected_statuses,
        selected_metrics=selected_metrics,
        target_value=target_value,
        watched_best_statuses=list(watched_best),
    )


def expand_status_names(base_metrics: list[str]) -> list[str]:
    expanded = list(base_metrics)
    for metric_name in base_metrics:
        expanded.extend(
            f'{metric_name}_{suffix}'
            for suffix in STATUS_ONLY_DERIVED_STATUS_RESOLVERS
        )
    return expanded


def required_metrics(selected_statuses: list[str]) -> list[str]:
    metrics: list[str] = []
    for status_name in selected_statuses:
        metric_name = base_metric_name(status_name)
        if metric_name not in metrics:
            metrics.append(metric_name)
    return metrics


def anchor_required_metrics(selected_statuses: list[str]) -> list[str]:
    metrics: list[str] = []
    for status_name in selected_statuses:
        if base_metric_name(status_name) == status_name:
            continue
        metric_name = base_metric_name(status_name)
        if metric_name not in metrics:
            metrics.append(metric_name)
    return metrics


def base_metric_name(status_name: str) -> str:
    for suffix in STATUS_ONLY_DERIVED_STATUS_RESOLVERS:
        token = f'_{suffix}'
        if status_name.endswith(token):
            return status_name.removesuffix(token)
    return status_name


def resolve_status_value(
    status_name: str,
    *,
    current_metrics: Mapping[str, float],
    anchor_metrics: Mapping[str, float],
) -> float:
    if status_name in current_metrics:
        return current_metrics[status_name]

    metric_name = base_metric_name(status_name)
    if metric_name == status_name:
        msg = f'Unknown status name: {status_name}'
        raise ValueError(msg)

    suffix = status_name.removeprefix(f'{metric_name}_')
    resolver = STATUS_ONLY_DERIVED_STATUS_RESOLVERS.get(suffix)
    if resolver is None:
        msg = f'Unknown status name: {status_name}'
        raise ValueError(msg)
    return resolver(
        metric_name,
        current_metrics=current_metrics,
        anchor_metrics=anchor_metrics,
    )


def format_status_line(
    status_values: Mapping[str, float],
    *,
    selected_statuses: list[str],
) -> Text:
    line = Text()
    line.append('status', style='dim')
    line.append(': ', style='dim')
    first = True
    for name in selected_statuses:
        if not first:
            line.append(' | ', style='dim')
        first = False
        line.append(name, style='cyan')
        line.append('=', style='dim')
        line.append_text(format_status_value(name, status_values[name]))
    return line


def format_watched_value_line(
    metric_name: str,
    *,
    current: float,
    best: float,
) -> Text:
    line = Text()
    line.append('watched value', style='dim')
    line.append(' ', style='dim')
    line.append(f'[{metric_name}]', style='cyan')
    line.append(': ', style='dim')
    line.append('current', style='cyan')
    line.append('=', style='dim')
    line.append_text(format_status_value(metric_name, current))
    line.append(' | ', style='dim')
    line.append('best', style='cyan')
    line.append('=', style='dim')
    line.append_text(format_status_value(metric_name, best))
    return line


def format_best_values_line(
    best_values: Mapping[str, float],
    *,
    selected_statuses: list[str],
) -> Text:
    line = Text()
    line.append('best values', style='dim')
    line.append(': ', style='dim')
    first = True
    for name in selected_statuses:
        if not first:
            line.append(' | ', style='dim')
        first = False
        line.append(name, style='cyan')
        line.append('=', style='dim')
        line.append_text(format_status_value(name, best_values[name]))
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
            rendered.stylize('green')
            return rendered
        if value < 0:
            rendered.stylize('red')
            return rendered
    rendered.stylize('white')
    return rendered


def ensure_status_runtime_available(
    status_config: ResolvedStatusConfig,
) -> None:
    if 'vmaf' in status_config.selected_metrics:
        ensure_vmaf_runtime_available()


def _resolve_gain_status(
    metric_name: str,
    *,
    current_metrics: Mapping[str, float],
    anchor_metrics: Mapping[str, float],
) -> float:
    return current_metrics[metric_name] - anchor_metrics[metric_name]


STATUS_ONLY_DERIVED_STATUS_RESOLVERS: dict[
    str,
    Callable[..., float],
] = {
    'gain': _resolve_gain_status,
}
