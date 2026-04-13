from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True, slots=True)
class ImagePair:
    source_path: Path
    target_path: Path


@dataclass(frozen=True, slots=True)
class CachedImagePair:
    source: torch.Tensor
    target: torch.Tensor


@dataclass(frozen=True, slots=True)
class DatasetSample:
    pair_index: int
    top: int | None = None
    left: int | None = None
    size: int | None = None


class ImageRestorationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        pairs: list[CachedImagePair],
        samples: list[DatasetSample],
    ) -> None:
        self.pairs = pairs
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        pair = self.pairs[sample.pair_index]
        if sample.top is None or sample.left is None or sample.size is None:
            return pair.source, pair.target

        top = sample.top
        left = sample.left
        bottom = top + sample.size
        right = left + sample.size
        return (
            pair.source[:, top:bottom, left:right],
            pair.target[:, top:bottom, left:right],
        )
