from pathlib import Path

import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

from .config import OnnxExportConfig, color_mode_channels
from .onnx_export_setup import export_dtype, load_export_checkpoint


class ExportSpec:
    def __init__(
        self,
        *,
        output_path: Path,
        in_channels: int,
        height: int,
        width: int,
        opset: int,
    ) -> None:
        self.output_path = output_path
        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.opset = opset


def export_onnx_model(config: OnnxExportConfig) -> Path:
    loaded = load_export_checkpoint(config)
    model = loaded.model.cpu()
    export_spec = ExportSpec(
        output_path=loaded.output_path,
        in_channels=color_mode_channels(loaded.color_mode),
        height=config.height,
        width=config.width,
        opset=config.opset,
    )

    if config.height <= 0 or config.width <= 0:
        msg = 'Export height and width must be positive.'
        raise ValueError(msg)
    if config.opset <= 0:
        msg = 'Export opset must be positive.'
        raise ValueError(msg)

    if loaded.precision == 'int8':
        _export_int8_onnx(
            model=model,
            export_spec=export_spec,
        )
    else:
        _export_typed_onnx(
            model=model.to(dtype=export_dtype(loaded.precision)),
            export_spec=export_spec,
            dtype=export_dtype(loaded.precision),
        )

    onnx_model = onnx.load(export_spec.output_path)
    onnx.checker.check_model(onnx_model)
    return export_spec.output_path


def _export_typed_onnx(
    *,
    model: torch.nn.Module,
    export_spec: ExportSpec,
    dtype: torch.dtype,
) -> None:
    dummy_input = torch.zeros(
        1,
        export_spec.in_channels,
        export_spec.height,
        export_spec.width,
        dtype=dtype,
    )
    torch.onnx.export(
        model,
        (dummy_input,),
        export_spec.output_path,
        export_params=True,
        opset_version=export_spec.opset,
        do_constant_folding=True,
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
                output_path=intermediate_path,
                in_channels=export_spec.in_channels,
                height=export_spec.height,
                width=export_spec.width,
                opset=export_spec.opset,
            ),
            dtype=torch.float32,
        )
        quantize_dynamic(
            intermediate_path,
            export_spec.output_path,
            weight_type=QuantType.QInt8,
        )
    finally:
        intermediate_path.unlink(missing_ok=True)
