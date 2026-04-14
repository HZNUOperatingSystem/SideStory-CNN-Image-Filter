from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .checkpoint import resolve_run_checkpoint_path


@dataclass(frozen=True, slots=True)
class ResolvedCheckpointCommand:
    checkpoint_path: Path
    output_path: Path


@dataclass(frozen=True, slots=True)
class CheckpointCommandPolicy:
    default_output: Callable[[Path], Path]
    output_conflict_message: str
    output_required_message: str


def resolve_checkpoint_command(
    *,
    run_dir: Path | None,
    ckpt: Path | None,
    output: Path | None,
    policy: CheckpointCommandPolicy,
) -> ResolvedCheckpointCommand:
    checkpoint_path = resolve_run_checkpoint_path(run_dir=run_dir, ckpt=ckpt)
    if run_dir is not None:
        if output is not None:
            raise ValueError(policy.output_conflict_message)
        return ResolvedCheckpointCommand(
            checkpoint_path=checkpoint_path,
            output_path=policy.default_output(run_dir),
        )

    if output is None:
        raise ValueError(policy.output_required_message)
    return ResolvedCheckpointCommand(
        checkpoint_path=checkpoint_path,
        output_path=output,
    )
