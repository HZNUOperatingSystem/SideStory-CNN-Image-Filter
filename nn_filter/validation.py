from dataclasses import dataclass

import torch
from rich.text import Text
from torch import nn
from torch.utils.data import DataLoader

from .config import ColorMode, StatusSelection
from .status import StatusTracker
from .ui import progress


@dataclass(slots=True)
class ValidationSummary:
    loss: float
    status_line: Text | None = None


class Validator:
    def __init__(
        self,
        criterion: nn.Module,
        *,
        color_mode: ColorMode,
        status: StatusSelection = False,
    ) -> None:
        self.criterion = criterion
        self.status_tracker = StatusTracker(status, color_mode=color_mode)

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

        return ValidationSummary(
            loss=total_loss / len(loader),
            status_line=self.status_tracker.finish_epoch(),
        )
