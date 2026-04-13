from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

import torch
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset

from .config import TrainConfig, color_mode_channels
from .data_setup import create_dataset
from .model import CNNFilter
from .runs import RunManager
from .status import ensure_status_runtime_available, resolve_status_config
from .ui import (
    print_batching_adjustment,
    print_dataset_summary,
    print_device,
    print_epoch_summary,
    print_text,
    progress,
)
from .validation import Validator


class _ClosableQueue(Protocol):
    def cancel_join_thread(self) -> None: ...

    def close(self) -> None: ...


class _TerminableWorker(Protocol):
    def is_alive(self) -> bool: ...

    def terminate(self) -> None: ...

    def join(self, timeout: float | None = None) -> None: ...

    def kill(self) -> None: ...


class _TrainState(Protocol):
    optimizer: torch.optim.Optimizer
    scheduler: OneCycleLR
    criterion: nn.Module


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    train_state: _TrainState,
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
        train_batch_size = _resolve_batch_size(
            requested_batch_size=config.batch_size,
            has_mixed_resolution=train_summary.has_mixed_resolution,
            patch_size=config.patch_size,
        )
        val_batch_size = 1
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
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-4,
        )
        scheduler = _build_scheduler(
            optimizer=optimizer,
            steps_per_epoch=len(train_loader),
            epochs=config.epochs,
            lr=config.lr,
            lr_min=config.lr_min,
        )
        train_state = _build_train_state(
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
        )
        validator = Validator(
            criterion,
            status_config=status_config,
        )

        for epoch in range(config.epochs):
            train_loss = train_epoch(
                model,
                train_loader,
                train_state,
                training_device,
            )
            validation = validator.evaluate(
                model, val_loader, training_device
            )
            print_epoch_summary(
                epoch=epoch + 1,
                total_epochs=config.epochs,
                train_loss=train_loss,
                val_loss=validation.loss,
                lr=optimizer.param_groups[0]['lr'],
            )
            epoch_record = run.record_epoch(
                model=model,
                model_config=model_config,
                epoch=epoch + 1,
                validation=validation,
            )
            for line in epoch_record.lines:
                print_text(line)
    finally:
        _shutdown_loader_workers(train_loader)
        _shutdown_loader_workers(val_loader)
        run.close()
    return run.run_dir


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


def _build_scheduler(
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
        anneal_strategy='cos',
        div_factor=div_factor,
        final_div_factor=initial_lr / lr_min,
    )


def _build_train_state(
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: OneCycleLR,
    criterion: nn.Module,
) -> _TrainState:
    class TrainState:
        def __init__(
            self,
            *,
            optimizer: torch.optim.Optimizer,
            scheduler: OneCycleLR,
            criterion: nn.Module,
        ) -> None:
            self.optimizer = optimizer
            self.scheduler = scheduler
            self.criterion = criterion

    return TrainState(
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
    )


def _shutdown_loader_workers(loader: DataLoader | None) -> None:
    if loader is None:
        return
    iterator = getattr(loader, '_iterator', None)
    if iterator is None:
        return

    if hasattr(iterator, '_shutdown'):
        iterator._shutdown = True
    _close_queue(getattr(iterator, '_worker_result_queue', None))
    index_queues = getattr(iterator, '_index_queues', None)
    if index_queues is not None:
        for queue in index_queues:
            _close_queue(queue)
    _terminate_workers(getattr(iterator, '_workers', None))
    loader._iterator = None


def _close_queue(queue: _ClosableQueue | None) -> None:
    if queue is None:
        return
    cancel_join_thread = getattr(queue, 'cancel_join_thread', None)
    close = getattr(queue, 'close', None)
    if callable(cancel_join_thread):
        cancel_join_thread()
    if callable(close):
        close()


def _terminate_workers(
    workers: Iterable[_TerminableWorker] | None,
) -> None:
    if workers is None:
        return
    for worker in workers:
        _terminate_worker(worker)


def _terminate_worker(worker: _TerminableWorker) -> None:
    if worker.is_alive():
        worker.terminate()
        worker.join(timeout=0.5)
    if worker.is_alive():
        worker.kill()
        worker.join(timeout=0.5)
