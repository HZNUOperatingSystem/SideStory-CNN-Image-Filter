from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import ImageRestorationDataset
from .model import CNNFilter


@dataclass(slots=True)
class TrainConfig:
    low_dir: Path
    high_dir: Path
    epochs: int = 10
    batch_size: int = 8
    lr: float = 1e-3
    save_dir: Path = Path('checkpoints')
    num_workers: int = 2


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

    train_dataset = ImageRestorationDataset(
        str(config.low_dir), str(config.high_dir)
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )

    model = CNNFilter().to(training_device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    save_dir = config.save_dir
    save_dir.mkdir(exist_ok=True)

    for epoch in range(config.epochs):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, training_device
        )
        print(f'Epoch {epoch + 1}/{config.epochs}, Loss: {train_loss:.4f}')
        torch.save(model.state_dict(), save_dir / f'epoch_{epoch + 1}.pt')
