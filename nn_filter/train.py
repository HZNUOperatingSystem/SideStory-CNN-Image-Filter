from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import TrainConfig, color_mode_channels
from .data_setup import create_dataset
from .model import CNNFilter


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


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for low_batch, high_batch in tqdm(loader, desc='val'):
            low, high = low_batch.to(device), high_batch.to(device)
            pred = model(low)
            loss = criterion(pred, high)
            total_loss += loss.item()
    return total_loss / len(loader)


def train_model(
    config: TrainConfig, *, device: torch.device | None = None
) -> None:
    training_device = device if device is not None else get_device()
    print(f'Using device: {training_device}')

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
    print(
        'Loaded datasets: '
        f'train={len(train_dataset)}, val={len(val_dataset)}, '
        f'image_size={train_image_size}, '
        f'color_mode={config.color_mode}, '
        f'patch_size={config.patch_size}'
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

    model = CNNFilter(
        in_channels=color_mode_channels(config.color_mode)
    ).to(training_device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    save_dir = config.save_dir
    save_dir.mkdir(exist_ok=True)

    for epoch in range(config.epochs):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, training_device
        )
        val_loss = validate(model, val_loader, criterion, training_device)
        print(
            f'Epoch {epoch + 1}/{config.epochs}, '
            f'train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}'
        )
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
