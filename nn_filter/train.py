from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import TrainConfig, color_mode_channels
from .data_setup import create_dataset
from .loader_utils import shutdown_loader_workers
from .model import CNNFilter
from .runs import EpochTrainingState, RunManager
from .runtime import get_device, set_seed
from .status import ensure_status_runtime_available, resolve_status_config
from .training_setup import (
    build_training_components,
    build_training_loaders,
)
from .ui import (
    print_batching_adjustment,
    print_dataset_summary,
    print_device,
    print_epoch_summary,
    print_text,
    progress,
)
from .validation import Validator


@dataclass(frozen=True, slots=True)
class TrainStepState:
    criterion: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.OneCycleLR


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    train_state: TrainStepState,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    sample_count = 0
    for low_batch, high_batch in progress(loader, desc='train'):
        low, high = low_batch.to(device), high_batch.to(device)
        train_state.optimizer.zero_grad(set_to_none=True)
        pred = model(low)
        loss = train_state.criterion(pred, high)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        train_state.optimizer.step()
        train_state.scheduler.step()
        total_loss += loss.item() * low.shape[0]
        sample_count += low.shape[0]
    return total_loss / sample_count


def train_model(
    config: TrainConfig, *, device: torch.device | None = None
) -> Path:
    set_seed(config.seed)
    training_device = device if device is not None else get_device()
    train_loader: DataLoader | None = None
    val_loader: DataLoader | None = None
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
            build_mode=(
                'patch-grid' if config.patch_size is not None else 'full-image'
            ),
        )
        val_dataset, val_summary = create_dataset(
            config.val_manifest,
            color_mode=config.color_mode,
            build_mode='full-image',
        )

        model_config = {'color_mode': config.color_mode}
        loaders = build_training_loaders(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=config,
            train_has_mixed_resolution=train_summary.has_mixed_resolution,
            device=training_device,
        )
        print_dataset_summary(
            train_summary=(
                f'train={len(train_dataset)} '
                f'(size={train_summary.image_size_label})'
            ),
            val_summary=(
                f'val={len(val_dataset)} (size={val_summary.image_size_label})'
            ),
            color_mode=config.color_mode,
            patch_size=config.patch_size,
        )
        if loaders.train_batch_size != config.batch_size:
            print_batching_adjustment(
                split='train',
                requested_batch_size=config.batch_size,
                actual_batch_size=loaders.train_batch_size,
            )
        if loaders.val_batch_size != config.batch_size:
            print_batching_adjustment(
                split='val',
                requested_batch_size=config.batch_size,
                actual_batch_size=loaders.val_batch_size,
            )

        train_loader = loaders.train_loader
        val_loader = loaders.val_loader

        model = CNNFilter(
            in_channels=color_mode_channels(config.color_mode)
        ).to(training_device)
        components = build_training_components(
            model,
            lr=config.lr,
            lr_min=config.lr_min,
            steps_per_epoch=len(train_loader),
            epochs=config.epochs,
        )
        train_state = TrainStepState(
            criterion=components.criterion,
            optimizer=components.optimizer,
            scheduler=components.scheduler,
        )
        validator = Validator(
            components.criterion,
            status_config=status_config,
        )

        for epoch in range(config.epochs):
            train_loss = train_epoch(
                model,
                train_loader,
                train_state,
                device=training_device,
            )
            validation = validator.evaluate(model, val_loader, training_device)
            print_epoch_summary(
                epoch=epoch + 1,
                total_epochs=config.epochs,
                train_loss=train_loss,
                val_loss=validation.loss,
                lr=components.optimizer.param_groups[0]['lr'],
            )
            epoch_record = run.record_epoch(
                model=model,
                model_config=model_config,
                epoch_state=EpochTrainingState(
                    epoch=epoch + 1,
                    train_loss=train_loss,
                    lr=components.optimizer.param_groups[0]['lr'],
                ),
                validation=validation,
            )
            for line in epoch_record.lines:
                print_text(line)
    finally:
        close_error: BaseException | None = None
        try:
            run.close()
        except BaseException as error:
            close_error = error
        finally:
            shutdown_loader_workers(train_loader)
            shutdown_loader_workers(val_loader)
        if close_error is not None:
            raise close_error
    return run.run_dir
