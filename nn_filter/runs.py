from __future__ import annotations

import tarfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from rich.text import Text
from torch import nn

from .status import (
    ResolvedStatusConfig,
    format_best_values_line,
    format_status_line,
    format_watched_value_line,
)
from .ui import print_run_directory, save_terminal_log
from .validation import ValidationSummary

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(slots=True)
class EpochRecord:
    lines: list[Text]


class RunManager:
    def __init__(
        self,
        run_dir: Path,
        *,
        status_config: ResolvedStatusConfig,
    ) -> None:
        self.run_dir = run_dir
        self.status_config = status_config
        self.value_name = (
            status_config.target_value
            if status_config.target_value is not None
            else 'val_loss'
        )
        self.best_value: float | None = None

    @classmethod
    def open(
        cls,
        runs_dir: Path,
        *,
        status_config: ResolvedStatusConfig,
    ) -> RunManager:
        run_dir = _create_run_directory(runs_dir)
        _write_source_archive(
            source_dir=PROJECT_ROOT / 'nn_filter',
            archive_path=run_dir / 'nn_filter.tar',
        )
        print_run_directory(run_dir)
        return cls(run_dir, status_config=status_config)

    def record_epoch(
        self,
        *,
        model: nn.Module,
        model_config: Mapping[str, str],
        epoch: int,
        validation: ValidationSummary,
    ) -> EpochRecord:
        value = self._resolve_value(validation)
        best_value = self._update_best_value(value)
        checkpoint_data = self._checkpoint_data(
            model=model,
            model_config=model_config,
            epoch=epoch,
            value=value,
            best_value=best_value,
        )

        self._save_checkpoint(self.run_dir / 'last.pt', checkpoint_data)
        if self._is_best(value):
            self._save_checkpoint(self.run_dir / 'best.pt', checkpoint_data)

        lines: list[Text] = []
        if self.status_config.target_value is not None:
            target_value = self.status_config.target_value
            lines.append(
                format_watched_value_line(
                    target_value,
                    current=validation.current_metrics[target_value],
                    best=best_value,
                )
            )
        if self.status_config.watched_best_statuses:
            lines.append(
                format_best_values_line(
                    validation.best_status_values,
                    selected_statuses=self.status_config.watched_best_statuses,
                )
            )
        if self.status_config.selected_metrics:
            lines.append(
                format_status_line(
                    validation.status_values,
                    selected_statuses=self.status_config.selected_statuses,
                )
            )
        return EpochRecord(lines=lines)

    def close(self) -> None:
        save_terminal_log(self.run_dir / 'terminal.log')

    def _resolve_value(self, validation: ValidationSummary) -> float:
        if self.value_name == 'val_loss':
            return validation.loss
        return validation.current_metrics[self.value_name]

    def _update_best_value(self, value: float) -> float:
        if self.best_value is None or self._is_better(value, self.best_value):
            self.best_value = value
        return self.best_value

    def _is_best(self, value: float) -> bool:
        return self.best_value == value

    def _is_better(self, candidate: float, current: float) -> bool:
        if self.value_name == 'val_loss':
            return candidate < current
        return candidate > current

    def _save_checkpoint(
        self,
        checkpoint_path: Path,
        checkpoint_data: Mapping[str, object],
    ) -> None:
        torch.save(checkpoint_data, checkpoint_path)

    def _checkpoint_data(
        self,
        *,
        model: nn.Module,
        model_config: Mapping[str, str],
        epoch: int,
        value: float,
        best_value: float,
    ) -> dict[str, object]:
        return {
            'epoch': epoch,
            'model_config': model_config,
            'model_state_dict': model.state_dict(),
            'value_name': self.value_name,
            'value': value,
            'best_value': best_value,
        }


def _create_run_directory(runs_dir: Path) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    next_index = 1
    numeric_dirs = [
        int(path.name)
        for path in runs_dir.iterdir()
        if path.is_dir() and path.name.isdigit()
    ]
    if numeric_dirs:
        next_index = max(numeric_dirs) + 1

    run_dir = runs_dir / str(next_index)
    run_dir.mkdir()
    return run_dir


def _write_source_archive(source_dir: Path, *, archive_path: Path) -> None:
    with tarfile.open(archive_path, 'w') as archive:
        archive.add(
            source_dir,
            arcname=source_dir.name,
            filter=_archive_filter,
        )


def _archive_filter(
    tar_info: tarfile.TarInfo,
) -> tarfile.TarInfo | None:
    if '__pycache__' in Path(tar_info.name).parts:
        return None
    return tar_info
