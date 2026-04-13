from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from .status import ResolvedStatusConfig, StatusTracker
from .ui import progress


@dataclass(slots=True)
class ValidationSummary:
    loss: float
    current_metrics: dict[str, float]
    best_metrics: dict[str, float]
    status_values: dict[str, float]


class Validator:
    def __init__(
        self,
        criterion: nn.Module,
        *,
        status_config: ResolvedStatusConfig,
    ) -> None:
        self.criterion = criterion
        self.status_tracker = StatusTracker(status_config)

    def evaluate(
        self,
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
    ) -> ValidationSummary:
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for low_batch, high_batch in progress(loader, desc='val'):
                low = low_batch.to(device)
                high = high_batch.to(device)
                prediction = model(low)
                loss = self.criterion(prediction, high)
                total_loss += loss.item()
                self.status_tracker.update(
                    prediction.detach().cpu(),
                    high.detach().cpu(),
                    batch_size=low.shape[0],
                )
        status_summary = self.status_tracker.finish_epoch()

        return ValidationSummary(
            loss=total_loss / len(loader),
            current_metrics=status_summary.current_metrics,
            best_metrics=status_summary.best_metrics,
            status_values=status_summary.status_values,
        )
