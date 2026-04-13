from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ImageRestorationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, low_dir: str, high_dir: str):
        self.low_dir = Path(low_dir)
        self.high_dir = Path(high_dir)
        self.samples = sorted([p.name for p in self.low_dir.glob('*')])
        self.transform = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        name = self.samples[index]
        low = Image.open(self.low_dir / name).convert('RGB')
        high = Image.open(self.high_dir / name).convert('RGB')
        return self.transform(low), self.transform(high)
