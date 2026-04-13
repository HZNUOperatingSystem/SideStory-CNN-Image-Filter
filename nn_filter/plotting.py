from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots


@dataclass(frozen=True, slots=True)
class EpochPlotRecord:
    epoch: int
    train_loss: float
    val_loss: float
    lr: float
    current_metrics: Mapping[str, float]
    best_metrics: Mapping[str, float]
    status_values: Mapping[str, float]
    best_status_values: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class MetricPlotSeries:
    name: str
    current_values: list[float]
    best_values: list[float]


@dataclass(frozen=True, slots=True)
class PlotPanelSpec:
    title: str
    axis_type: str


def write_training_metrics_plot(
    records: Sequence[EpochPlotRecord],
    *,
    output_path: Path,
) -> None:
    if not records:
        return

    metric_names = _ordered_names(
        [record.current_metrics for record in records]
    )
    derived_status_names = _ordered_names(
        [
            {
                name: value
                for name, value in record.status_values.items()
                if name not in record.current_metrics
            }
            for record in records
        ]
    )
    panel_specs = _build_panel_specs(metric_names, derived_status_names)
    row_count = (len(panel_specs) + 1) // 2
    figure = make_subplots(
        rows=row_count,
        cols=2,
        shared_xaxes=False,
        vertical_spacing=0.10,
        horizontal_spacing=0.10,
        subplot_titles=_subplot_titles(panel_specs, row_count=row_count),
    )

    epochs = [record.epoch for record in records]
    _add_loss_traces(figure, records, epochs=epochs, row=1, col=1)
    _add_lr_trace(figure, records, epochs=epochs, row=1, col=2)

    panel_index = 2
    for metric_name in metric_names:
        row, col = _grid_position(panel_index)
        _add_metric_series(
            figure,
            epochs=epochs,
            row=row,
            col=col,
            series=_build_metric_series(
                records,
                name=metric_name,
                current_key='current_metrics',
                best_key='best_metrics',
            ),
        )
        panel_index += 1
    for status_name in derived_status_names:
        row, col = _grid_position(panel_index)
        _add_metric_series(
            figure,
            epochs=epochs,
            row=row,
            col=col,
            series=_build_metric_series(
                records,
                name=status_name,
                current_key='status_values',
                best_key='best_status_values',
            ),
        )
        panel_index += 1

    for panel_index, panel_spec in enumerate(panel_specs):
        row, col = _grid_position(panel_index)
        if panel_spec.axis_type == 'log':
            figure.update_yaxes(type='log', row=row, col=col)

    figure.update_layout(
        template='plotly_white',
        title='Training Metrics',
        width=1280,
        height=max(960, row_count * 420),
        showlegend=False,
        margin={'l': 70, 'r': 50, 't': 120, 'b': 70},
    )
    figure.update_xaxes(title_text='epoch', row=row_count, col=1)
    figure.update_xaxes(title_text='epoch', row=row_count, col=2)
    figure.update_annotations(font={'size': 15})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(
        output_path,
        include_plotlyjs='cdn',
        full_html=True,
    )


def _ordered_names(mappings: Iterable[Mapping[str, float]]) -> list[str]:
    ordered: list[str] = []
    for mapping in mappings:
        for name in mapping:
            if name not in ordered:
                ordered.append(name)
    return ordered


def _build_panel_specs(
    metric_names: list[str],
    derived_status_names: list[str],
) -> list[PlotPanelSpec]:
    panels = [
        PlotPanelSpec('loss [train, val]', 'log'),
        PlotPanelSpec('lr', 'log'),
    ]
    panels.extend(
        PlotPanelSpec(f'{name} [current, best]', 'linear')
        for name in metric_names
    )
    panels.extend(
        PlotPanelSpec(f'{name} [current, best]', 'linear')
        for name in derived_status_names
    )
    return panels


def _subplot_titles(
    panel_specs: list[PlotPanelSpec],
    *,
    row_count: int,
) -> list[str]:
    titles = [panel.title for panel in panel_specs]
    target_count = row_count * 2
    while len(titles) < target_count:
        titles.append('')
    return titles


def _grid_position(panel_index: int) -> tuple[int, int]:
    row = panel_index // 2 + 1
    col = panel_index % 2 + 1
    return row, col


def _add_loss_traces(
    figure: go.Figure,
    records: Sequence[EpochPlotRecord],
    *,
    epochs: list[int],
    row: int,
    col: int,
) -> None:
    train_loss = [record.train_loss for record in records]
    val_loss = [record.val_loss for record in records]
    figure.add_trace(
        go.Scatter(
            x=epochs,
            y=train_loss,
            mode='lines+markers',
            name='train_loss',
            line={'color': '#16a34a', 'width': 2},
            marker={'size': 6},
        ),
        row=row,
        col=col,
    )
    figure.add_trace(
        go.Scatter(
            x=epochs,
            y=val_loss,
            mode='lines+markers',
            name='val_loss',
            line={'color': '#f59e0b', 'width': 2},
            marker={'size': 6},
        ),
        row=row,
        col=col,
    )


def _add_lr_trace(
    figure: go.Figure,
    records: Sequence[EpochPlotRecord],
    *,
    epochs: list[int],
    row: int,
    col: int,
) -> None:
    lr_values = [record.lr for record in records]
    figure.add_trace(
        go.Scatter(
            x=epochs,
            y=lr_values,
            mode='lines+markers',
            name='lr',
            line={'color': '#a855f7', 'width': 2},
            marker={'size': 6},
        ),
        row=row,
        col=col,
    )


def _build_metric_series(
    records: Sequence[EpochPlotRecord],
    *,
    name: str,
    current_key: str,
    best_key: str,
) -> MetricPlotSeries:
    current_values = [
        _record_mapping(record, current_key).get(name, math.nan)
        for record in records
    ]
    best_values = [
        _record_mapping(record, best_key).get(name, math.nan)
        for record in records
    ]
    return MetricPlotSeries(
        name=name,
        current_values=current_values,
        best_values=best_values,
    )


def _add_metric_series(
    figure: go.Figure,
    *,
    epochs: list[int],
    row: int,
    col: int,
    series: MetricPlotSeries,
) -> None:
    figure.add_trace(
        go.Scatter(
            x=epochs,
            y=series.current_values,
            mode='lines+markers',
            name=series.name,
            line={'width': 2},
            marker={'size': 6},
        ),
        row=row,
        col=col,
    )
    if _has_distinct_best_series(
        series.current_values, series.best_values
    ):
        figure.add_trace(
            go.Scatter(
                x=epochs,
                y=series.best_values,
                mode='lines',
                name=f'best_{series.name}',
                line={'width': 2, 'dash': 'dash'},
            ),
            row=row,
            col=col,
        )


def _record_mapping(
    record: EpochPlotRecord,
    key: str,
) -> Mapping[str, float]:
    return getattr(record, key)


def _has_distinct_best_series(
    current_values: Sequence[float],
    best_values: Sequence[float],
) -> bool:
    if len(current_values) != len(best_values):
        return True
    return any(
        not math.isclose(current, best, rel_tol=1e-9, abs_tol=1e-12)
        for current, best in zip(current_values, best_values, strict=True)
        if not math.isnan(current) and not math.isnan(best)
    )
