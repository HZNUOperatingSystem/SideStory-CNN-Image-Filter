from rich.console import Console
from rich.text import Text

console = Console()


def print_device(device: object) -> None:
    console.print(
        Text.assemble(
            ('device', 'bold cyan'),
            ': ',
            (str(device), 'bold green'),
        )
    )


def print_dataset_summary(
    *,
    train_count: int,
    val_count: int,
    image_size: tuple[int, int],
    color_mode: str,
    patch_size: int | None,
) -> None:
    summary = Text()
    summary.append('dataset', style='bold cyan')
    summary.append(': ')
    summary.append(f'train={train_count}', style='green')
    summary.append(' | ')
    summary.append(f'val={val_count}', style='green')
    summary.append(' | ')
    summary.append(f'image_size={image_size}', style='yellow')
    summary.append(' | ')
    summary.append(f'color_mode={color_mode}', style='magenta')
    summary.append(' | ')
    summary.append(f'patch_size={patch_size}', style='blue')
    console.print(summary)


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
