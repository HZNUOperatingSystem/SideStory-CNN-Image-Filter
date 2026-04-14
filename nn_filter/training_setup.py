from dataclasses import dataclass

import torch
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset

from .config import TrainConfig


@dataclass(frozen=True, slots=True)
class TrainingLoaders:
    train_loader: DataLoader
    val_loader: DataLoader
    train_batch_size: int
    val_batch_size: int


@dataclass(frozen=True, slots=True)
class TrainingComponents:
    criterion: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: OneCycleLR


def build_training_loaders(
    *,
    train_dataset: Dataset[tuple[torch.Tensor, torch.Tensor]],
    val_dataset: Dataset[tuple[torch.Tensor, torch.Tensor]],
    config: TrainConfig,
    train_has_mixed_resolution: bool,
    device: torch.device,
) -> TrainingLoaders:
    train_batch_size = resolve_batch_size(
        requested_batch_size=config.batch_size,
        has_mixed_resolution=train_has_mixed_resolution,
        patch_size=config.patch_size,
    )
    val_batch_size = 1
    return TrainingLoaders(
        train_loader=create_loader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            device=device,
        ),
        val_loader=create_loader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            device=device,
        ),
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size,
    )


def build_training_components(
    model: nn.Module,
    *,
    lr: float,
    lr_min: float,
    steps_per_epoch: int,
    epochs: int,
) -> TrainingComponents:
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-4,
    )
    scheduler = build_scheduler(
        optimizer=optimizer,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        lr=lr,
        lr_min=lr_min,
    )
    return TrainingComponents(
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
    )


def create_loader(
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


def resolve_batch_size(
    *,
    requested_batch_size: int,
    has_mixed_resolution: bool,
    patch_size: int | None,
) -> int:
    if patch_size is not None or not has_mixed_resolution:
        return requested_batch_size
    return 1


def build_scheduler(
    *,
    optimizer: torch.optim.Optimizer,
    steps_per_epoch: int,
    epochs: int,
    lr: float,
    lr_min: float,
) -> OneCycleLR:
    total_steps = steps_per_epoch * epochs
    if total_steps <= 0:
        msg = 'Training requires at least one optimization step.'
        raise ValueError(msg)
    div_factor = 10.0
    initial_lr = lr / div_factor
    if lr_min <= 0:
        msg = f'lr_min must be positive, got {lr_min}'
        raise ValueError(msg)
    if lr_min >= initial_lr:
        msg = (
            'lr_min must be smaller than the scheduler initial lr '
            f'({initial_lr:.2e}), got {lr_min:.2e}'
        )
        raise ValueError(msg)
    return OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=total_steps,
        pct_start=0.3,
        div_factor=div_factor,
        final_div_factor=initial_lr / lr_min,
        anneal_strategy='cos',
    )
