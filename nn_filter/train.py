import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import TrainConfig, color_mode_channels
from .data_setup import create_dataset
from .model import CNNFilter
from .runs import RunManager
from .status import ensure_status_runtime_available, resolve_status_config
from .ui import (
    console,
    print_batching_adjustment,
    print_dataset_summary,
    print_device,
    print_epoch_summary,
    progress,
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
    for low_batch, high_batch in progress(loader, desc='train'):
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
    status_config = resolve_status_config(
        config.status,
        target_value=config.target_value,
        watched_best=config.watched_best,
        color_mode=config.color_mode,
    )
    ensure_status_runtime_available(status_config)
    run = RunManager.open(config.runs_dir, status_config=status_config)
    try:
        print_device(training_device)

        train_dataset, train_summary = create_dataset(
            config.train_manifest,
            color_mode=config.color_mode,
            patch_size=config.patch_size,
            random_crop=True,
        )
        val_dataset, val_summary = create_dataset(
            config.val_manifest,
            color_mode=config.color_mode,
            patch_size=config.patch_size,
        )

        model_config = {'color_mode': config.color_mode}
        train_batch_size = _resolve_batch_size(
            requested_batch_size=config.batch_size,
            has_mixed_resolution=train_summary.has_mixed_resolution,
            patch_size=config.patch_size,
        )
        val_batch_size = _resolve_batch_size(
            requested_batch_size=config.batch_size,
            has_mixed_resolution=val_summary.has_mixed_resolution,
            patch_size=config.patch_size,
        )
        print_dataset_summary(
            train_summary=(
                f'train={len(train_dataset)} '
                f'(size={train_summary.image_size_label})'
            ),
            val_summary=(
                f'val={len(val_dataset)} '
                f'(size={val_summary.image_size_label})'
            ),
            color_mode=config.color_mode,
            patch_size=config.patch_size,
        )
        if train_batch_size != config.batch_size:
            print_batching_adjustment(
                split='train',
                requested_batch_size=config.batch_size,
                actual_batch_size=train_batch_size,
            )
        if val_batch_size != config.batch_size:
            print_batching_adjustment(
                split='val',
                requested_batch_size=config.batch_size,
                actual_batch_size=val_batch_size,
            )

        train_loader = _create_loader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            device=training_device,
        )
        val_loader = _create_loader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            device=training_device,
        )

        model = CNNFilter(
            in_channels=color_mode_channels(config.color_mode)
        ).to(training_device)
        criterion = nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        validator = Validator(
            criterion,
            status_config=status_config,
        )

        for epoch in range(config.epochs):
            train_loss = train_epoch(
                model, train_loader, optimizer, criterion, training_device
            )
            validation = validator.evaluate(
                model, val_loader, training_device
            )
            print_epoch_summary(
                epoch=epoch + 1,
                total_epochs=config.epochs,
                train_loss=train_loss,
                val_loss=validation.loss,
            )
            epoch_record = run.record_epoch(
                model=model,
                model_config=model_config,
                epoch=epoch + 1,
                validation=validation,
            )
            for line in epoch_record.lines:
                console.print(line)
    finally:
        run.close()


def _create_loader(
    dataset: Dataset[tuple[torch.Tensor, torch.Tensor]],
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=device.type == 'cuda',
    )


def _resolve_batch_size(
    *,
    requested_batch_size: int,
    has_mixed_resolution: bool,
    patch_size: int | None,
) -> int:
    if patch_size is not None or not has_mixed_resolution:
        return requested_batch_size
    return 1
