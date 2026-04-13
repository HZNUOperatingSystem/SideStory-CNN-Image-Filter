from dataclasses import dataclass
from pathlib import Path

import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

from .config import ExportPrecision, OnnxExportConfig, color_mode_channels
from .onnx_export_setup import export_dtype, load_export_checkpoint


@dataclass(frozen=True, slots=True)
class ExportSpec:
    output_path: Path
    in_channels: int
    height: int
    width: int
    opset: int


def export_onnx_model(config: OnnxExportConfig) -> Path:
    loaded = load_export_checkpoint(config)
    export_spec = ExportSpec(
        loaded.output_path,
        color_mode_channels(loaded.color_mode),
        config.height,
        config.width,
        config.opset,
    )

    if config.height <= 0 or config.width <= 0:
        msg = 'Export height and width must be positive.'
        raise ValueError(msg)
    if config.opset <= 0:
        msg = 'Export opset must be positive.'
        raise ValueError(msg)

    if loaded.precision == 'int8':
        _export_int8_onnx(
            model=loaded.model.cpu(),
            export_spec=export_spec,
        )
    else:
        export_device = _select_export_device(loaded.precision)
        _export_typed_onnx(
            model=loaded.model.to(
                device=export_device,
                dtype=export_dtype(loaded.precision),
            ),
            export_spec=export_spec,
            dtype=export_dtype(loaded.precision),
            device=export_device,
            do_constant_folding=loaded.precision == 'fp32',
        )

    onnx_model = onnx.load(export_spec.output_path)
    onnx.checker.check_model(onnx_model)
    return export_spec.output_path


def _export_typed_onnx(
    *,
    model: torch.nn.Module,
    export_spec: ExportSpec,
    dtype: torch.dtype,
    device: torch.device,
    do_constant_folding: bool,
) -> None:
    dummy_input = torch.zeros(
        1,
        export_spec.in_channels,
        export_spec.height,
        export_spec.width,
        dtype=dtype,
        device=device,
    )
    torch.onnx.export(
        model,
        (dummy_input,),
        export_spec.output_path,
        export_params=True,
        opset_version=export_spec.opset,
        do_constant_folding=do_constant_folding,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch', 2: 'height', 3: 'width'},
            'output': {0: 'batch', 2: 'height', 3: 'width'},
        },
    )


def _export_int8_onnx(
    *,
    model: torch.nn.Module,
    export_spec: ExportSpec,
) -> None:
    intermediate_path = export_spec.output_path.with_suffix('.fp32.tmp.onnx')
    try:
        _export_typed_onnx(
            model=model,
            export_spec=ExportSpec(
                intermediate_path,
                export_spec.in_channels,
                export_spec.height,
                export_spec.width,
                export_spec.opset,
            ),
            dtype=torch.float32,
            device=torch.device('cpu'),
            do_constant_folding=True,
        )
        quantize_dynamic(
            intermediate_path,
            export_spec.output_path,
            weight_type=QuantType.QInt8,
        )
    finally:
        intermediate_path.unlink(missing_ok=True)


def _select_export_device(precision: ExportPrecision) -> torch.device:
    if precision in {'fp16', 'bf16'}:
        if torch.cuda.is_available():
            return torch.device('cuda')
        msg = f'{precision} ONNX export requires CUDA in this codebase.'
        raise RuntimeError(msg)
    return torch.device('cpu')
