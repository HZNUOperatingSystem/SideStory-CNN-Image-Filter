import argparse
from pathlib import Path

import torch
from dataset import ImageRestorationDataset
from model import CNNFilter
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
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
    for low, high in tqdm(loader, desc='train'):
        low, high = low.to(device), high.to(device)
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
        for low, high in tqdm(loader, desc='val'):
            low, high = low.to(device), high.to(device)
            pred = model(low)
            loss = criterion(pred, high)
            total_loss += loss.item()
    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--low_dir', required=True)
    parser.add_argument('--high_dir', required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save_dir', default='checkpoints')
    args = parser.parse_args()

    device = get_device()
    print(f'Using device: {device}')

    train_dataset = ImageRestorationDataset(args.low_dir, args.high_dir)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )

    model = CNNFilter().to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    for epoch in range(args.epochs):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        print(f'Epoch {epoch + 1}/{args.epochs}, Loss: {train_loss:.4f}')
        torch.save(model.state_dict(), save_dir / f'epoch_{epoch + 1}.pt')


if __name__ == '__main__':
    main()
