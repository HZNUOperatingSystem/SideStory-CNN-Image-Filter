from collections.abc import Iterable
from typing import TypeVar

from rich.console import Console
from rich.text import Text
from tqdm import tqdm

console = Console()
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


def print_device(device: object) -> None:
    console.print(
        Text.assemble(
            ('device', 'bold blue'),
            ': ',
            (str(device), 'bold green'),
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
    summary.append(' | ')
    summary.append(val_summary, style='yellow')
    summary.append(' | ')
    summary.append(f'color_mode={color_mode}', style='magenta')
    summary.append(' | ')
    summary.append(f'patch_size={patch_size}', style='blue')
    console.print(summary)


def print_batching_adjustment(
    *,
    split: str,
    requested_batch_size: int,
    actual_batch_size: int,
) -> None:
    console.print(
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
) -> None:
    summary = Text()
    summary.append(f'epoch {epoch}/{total_epochs}', style='bold cyan')
    summary.append(' | ')
    summary.append('train_loss=', style='white')
    summary.append(f'{train_loss:.4f}', style='bold green')
    summary.append(' | ')
    summary.append('val_loss=', style='white')
    summary.append(f'{val_loss:.4f}', style='bold yellow')
    console.print(summary)
