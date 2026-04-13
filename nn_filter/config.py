from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class TrainConfig:
    low_dir: Path
    high_dir: Path
    epochs: int = 10
    batch_size: int = 8
    lr: float = 1e-3
    save_dir: Path = Path('checkpoints')
    num_workers: int = 2
