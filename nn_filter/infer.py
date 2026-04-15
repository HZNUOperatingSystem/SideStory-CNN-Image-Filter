import torch

from .config import InferConfig
from .infer_setup import (
    load_checkpoint,
    load_inference_samples,
    resolve_infer_config,
)
from .io_utils import load_image_tensor, save_image_tensor
from .runtime import get_device
from .status import (
    DeviceStatusTracker,
    resolve_status_config,
)
from .status_ui import format_status_line
from .ui import print_device, print_infer_summary, print_text, progress


def infer_model(
    config: InferConfig, *, device: torch.device | None = None
) -> None:
    inference_device = device if device is not None else get_device()
    resolved_config = resolve_infer_config(config)
    loaded_checkpoint = load_checkpoint(
        resolved_config,
        device=inference_device,
    )
    status_config = resolve_status_config(
        resolved_config.status,
        target_value=None,
        watched_best=[],
        color_mode=loaded_checkpoint.color_mode,
    )
    input_path = resolved_config.input
    if input_path is None:
        msg = 'Input path could not be resolved.'
        raise ValueError(msg)
    samples = load_inference_samples(
        input_path,
        output_dir=loaded_checkpoint.output_dir,
    )
    status_tracker = DeviceStatusTracker(
        status_config,
        device=inference_device,
    )

    print_device(inference_device)
    for sample in progress(samples, desc='infer'):
        input_cpu = load_image_tensor(
            sample.input_path,
            color_mode=loaded_checkpoint.color_mode,
        )
        target_cpu = (
            load_image_tensor(
                sample.target_path,
                color_mode=loaded_checkpoint.color_mode,
            )
            if sample.target_path is not None
            else None
        )
        input_tensor = input_cpu.unsqueeze(0).to(inference_device)
        with torch.inference_mode():
            output_tensor = loaded_checkpoint.model(input_tensor)
        output_cpu = output_tensor.squeeze(0).detach().cpu()
        sample.output_path.parent.mkdir(parents=True, exist_ok=True)
        save_image_tensor(output_cpu, sample.output_path)
        if target_cpu is not None:
            target_tensor = target_cpu.unsqueeze(0).to(inference_device)
            status_tracker.update(
                prediction=output_tensor,
                target=target_tensor,
                anchor=input_tensor,
                batch_size=1,
            )

    print_infer_summary(
        checkpoint_path=loaded_checkpoint.checkpoint_path,
        output_dir=loaded_checkpoint.output_dir,
        sample_count=len(samples),
    )
    status_summary = status_tracker.finish_epoch()
    if status_summary.status_values:
        print_text(
            format_status_line(
                status_summary.status_values,
                selected_statuses=status_config.selected_statuses,
            )
        )
