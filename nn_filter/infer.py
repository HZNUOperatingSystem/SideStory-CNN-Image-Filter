import torch
from rich.text import Text

from .config import InferConfig
from .infer_setup import (
    load_checkpoint,
    load_inference_samples,
    load_inference_tensor,
)
from .io_utils import save_image_tensor
from .train import get_device
from .ui import print_device, print_text, progress


def infer_model(
    config: InferConfig, *, device: torch.device | None = None
) -> None:
    inference_device = device if device is not None else get_device()
    loaded_checkpoint = load_checkpoint(config, device=inference_device)
    samples = load_inference_samples(
        config.input,
        output_dir=loaded_checkpoint.output_dir,
    )

    print_device(inference_device)
    for sample in progress(samples, desc='infer'):
        input_tensor = load_inference_tensor(
            sample,
            color_mode=loaded_checkpoint.color_mode,
            device=inference_device,
        )
        with torch.no_grad():
            output_tensor = loaded_checkpoint.model(input_tensor)
        sample.output_path.parent.mkdir(parents=True, exist_ok=True)
        save_image_tensor(output_tensor.squeeze(0), sample.output_path)

    print_text(
        _build_infer_summary(
            checkpoint_path=loaded_checkpoint.checkpoint_path,
            output_dir=loaded_checkpoint.output_dir,
            sample_count=len(samples),
        )
    )


def _build_infer_summary(
    *,
    checkpoint_path: object,
    output_dir: object,
    sample_count: int,
) -> Text:
    summary = Text()
    summary.append('infer', style='bold blue')
    summary.append(': ')
    summary.append(str(checkpoint_path), style='green')
    summary.append(' | ', style='dim')
    summary.append(str(output_dir), style='yellow')
    summary.append(' | ', style='dim')
    summary.append(f'samples={sample_count}', style='magenta')
    return summary
