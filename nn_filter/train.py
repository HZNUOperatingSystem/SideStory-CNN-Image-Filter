import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import TrainConfig
from .dataset import ImageRestorationDataset
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

    train_dataset = ImageRestorationDataset(config.train_manifest)
    val_dataset = ImageRestorationDataset(config.val_manifest)
    if train_dataset.image_size != val_dataset.image_size:
        msg = (
            'Train and validation image sizes must match: '
            f'{train_dataset.image_size} vs {val_dataset.image_size}'
        )
        raise ValueError(msg)

    print(
        'Loaded datasets: '
        f'train={len(train_dataset)}, val={len(val_dataset)}, '
        f'image_size={train_dataset.image_size}'
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

    model = CNNFilter().to(training_device)
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
        torch.save(model.state_dict(), save_dir / f'epoch_{epoch + 1}.pt')
