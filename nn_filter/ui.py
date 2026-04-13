from collections.abc import Iterable
from pathlib import Path
from typing import TypeVar

from rich.console import Console
from rich.text import Text
from tqdm import tqdm

console = Console(record=True)
PROGRESS_COLOR = '#6b7280'
PROGRESS_ASCII = ' .-'
T = TypeVar('T')


def progress(iterable: Iterable[T], *, desc: str) -> Iterable[T]:
    return tqdm(
        iterable,
        desc=desc,
        leave=False,
        dynamic_ncols=True,
        colour=PROGRESS_COLOR,
        ascii=PROGRESS_ASCII,
    )


def print_text(text: Text) -> None:
    with tqdm.external_write_mode():
        console.print(text)


def print_device(device: object) -> None:
    print_text(
        Text.assemble(
            ('device', 'bold blue'),
            ': ',
            (str(device), 'bold green'),
        )
    )


def print_run_directory(run_dir: Path) -> None:
    print_text(
        Text.assemble(
            ('run', 'bold blue'),
            ': ',
            (str(run_dir), 'bold green'),
        )
    )


def print_dataset_summary(
    *,
    train_summary: str,
    val_summary: str,
    color_mode: str,
    patch_size: int | None,
) -> None:
    summary = Text()
    summary.append('dataset', style='bold blue')
    summary.append(': ')
    summary.append(train_summary, style='green')
    summary.append(' | ', style='dim')
    summary.append(val_summary, style='yellow')
    summary.append(' | ', style='dim')
    summary.append(f'color_mode={color_mode}', style='magenta')
    summary.append(' | ', style='dim')
    summary.append(f'patch_size={patch_size}', style='blue')
    print_text(summary)


def print_batching_adjustment(
    *,
    split: str,
    requested_batch_size: int,
    actual_batch_size: int,
) -> None:
    print_text(
        Text.assemble(
            ('batching', 'bold blue'),
            ': ',
            (
                f'{split} uses batch_size={actual_batch_size} ',
                'bold yellow',
            ),
            (
                f'(requested {requested_batch_size})',
                'yellow',
            ),
            ' because samples have mixed resolutions without patch extraction.',
        )
    )


def print_epoch_summary(
    *,
    epoch: int,
    total_epochs: int,
    train_loss: float,
    val_loss: float,
    lr: float | None = None,
) -> None:
    summary = Text()
    summary.append(f'epoch {epoch}/{total_epochs}', style='magenta')
    summary.append(' | ', style='dim')
    summary.append('train_loss', style='white')
    summary.append('=', style='dim')
    summary.append(f'{train_loss:.4f}', style='bold green')
    summary.append(' | ', style='dim')
    summary.append('val_loss', style='white')
    summary.append('=', style='dim')
    summary.append(f'{val_loss:.4f}', style='bold yellow')
    if lr is not None:
        summary.append(' | ', style='dim')
        summary.append('lr', style='white')
        summary.append('=', style='dim')
        summary.append(f'{lr:.2e}', style='magenta')
    print_text(summary)


def save_terminal_log(log_path: Path) -> None:
    log_path.write_text(
        console.export_text(styles=False, clear=False),
        encoding='utf-8',
    )
