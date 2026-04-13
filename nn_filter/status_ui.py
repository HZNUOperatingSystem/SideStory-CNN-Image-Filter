import math
from collections.abc import Mapping

from rich.text import Text


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
