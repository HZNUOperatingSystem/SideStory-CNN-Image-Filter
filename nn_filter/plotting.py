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
    row_specs = _build_row_specs(metric_names, derived_status_names)
    figure = make_subplots(
        rows=len(row_specs),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=[title for title, _ in row_specs],
    )

    epochs = [record.epoch for record in records]
    _add_loss_traces(figure, records, epochs=epochs, row=1)
    _add_lr_trace(figure, records, epochs=epochs, row=2)

    row_index = 3
    for metric_name in metric_names:
        _add_metric_series(
            figure,
            epochs=epochs,
            row=row_index,
            series=_build_metric_series(
                records,
                name=metric_name,
                current_key='current_metrics',
                best_key='best_metrics',
            ),
        )
        row_index += 1
    for status_name in derived_status_names:
        _add_metric_series(
            figure,
            epochs=epochs,
            row=row_index,
            series=_build_metric_series(
                records,
                name=status_name,
                current_key='status_values',
                best_key='best_status_values',
            ),
        )
        row_index += 1

    for row_number, (_, axis_type) in enumerate(row_specs, start=1):
        if axis_type == 'log':
            figure.update_yaxes(type='log', row=row_number, col=1)

    figure.update_layout(
        template='plotly_white',
        title='Training Metrics',
        width=1800,
        height=max(900, len(row_specs) * 320),
        legend={
            'orientation': 'h',
            'yanchor': 'bottom',
            'y': 1.02,
            'xanchor': 'left',
            'x': 0.0,
        },
        margin={'l': 80, 'r': 40, 't': 100, 'b': 70},
    )
    figure.update_xaxes(title_text='epoch', row=len(row_specs), col=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_image(output_path, scale=2)
    figure.write_html(
        output_path.with_suffix('.html'),
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


def _build_row_specs(
    metric_names: list[str],
    derived_status_names: list[str],
) -> list[tuple[str, str]]:
    rows = [('loss', 'log'), ('lr', 'log')]
    rows.extend((name, 'linear') for name in metric_names)
    rows.extend((name, 'linear') for name in derived_status_names)
    return rows


def _add_loss_traces(
    figure: go.Figure,
    records: Sequence[EpochPlotRecord],
    *,
    epochs: list[int],
    row: int,
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
        col=1,
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
        col=1,
    )


def _add_lr_trace(
    figure: go.Figure,
    records: Sequence[EpochPlotRecord],
    *,
    epochs: list[int],
    row: int,
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
        col=1,
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
        col=1,
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
            col=1,
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
