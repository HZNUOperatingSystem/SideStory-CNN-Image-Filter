from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import TrainConfig, color_mode_channels
from .data_setup import create_dataset
from .model import CNNFilter
from .ui import (
    console,
    print_dataset_summary,
    print_device,
    print_epoch_summary,
)
from .validation import Validator


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for low_batch, high_batch in tqdm(loader, desc='train'):
        low, high = low_batch.to(device), high_batch.to(device)
        optimizer.zero_grad()
        pred = model(low)
        loss = criterion(pred, high)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def train_model(
    config: TrainConfig, *, device: torch.device | None = None
) -> None:
    training_device = device if device is not None else get_device()
    print_device(training_device)

    train_dataset, train_image_size = create_dataset(
        config.train_manifest,
        color_mode=config.color_mode,
        patch_size=config.patch_size,
        random_crop=True,
    )
    val_dataset, val_image_size = create_dataset(
        config.val_manifest,
        color_mode=config.color_mode,
        patch_size=config.patch_size,
    )
    if train_image_size != val_image_size:
        msg = (
            'Train and validation image sizes must match: '
            f'{train_image_size} vs {val_image_size}'
        )
        raise ValueError(msg)

    model_config = {'color_mode': config.color_mode}
    print_dataset_summary(
        train_count=len(train_dataset),
        val_count=len(val_dataset),
        image_size=train_image_size,
        color_mode=config.color_mode,
        patch_size=config.patch_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model = CNNFilter(in_channels=color_mode_channels(config.color_mode)).to(
        training_device
    )
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    validator = Validator(
        criterion,
        color_mode=config.color_mode,
        status=config.status,
    )

    save_dir = config.save_dir
    save_dir.mkdir(exist_ok=True)

    for epoch in range(config.epochs):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, training_device
        )
        validation = validator.evaluate(model, val_loader, training_device)
        print_epoch_summary(
            epoch=epoch + 1,
            total_epochs=config.epochs,
            train_loss=train_loss,
            val_loss=validation.loss,
        )
        if validation.status_line is not None:
            console.print(validation.status_line)
        _save_checkpoint(
            save_dir / f'epoch_{epoch + 1}.pt',
            model=model,
            model_config=model_config,
            epoch=epoch + 1,
        )


def _save_checkpoint(
    checkpoint_path: Path,
    *,
    model: nn.Module,
    model_config: Mapping[str, str],
    epoch: int,
) -> None:
    torch.save(
        {
            'epoch': epoch,
            'model_config': model_config,
            'model_state_dict': model.state_dict(),
        },
        checkpoint_path,
    )
