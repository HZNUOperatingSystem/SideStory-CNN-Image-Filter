import argparse
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import cast

from nn_filter.config import InferConfig, OnnxExportConfig, TrainConfig

from .config import add_dataclass_arguments, load_config, namespace_overrides


def register_commands(subparsers: argparse._SubParsersAction) -> None:
    train_parser = subparsers.add_parser(
        'train',
        help='Train the image restoration model.',
        description='Train the image restoration model.',
    )
    train_parser.add_argument(
        '--config',
        type=Path,
        default=Path('configs/train.toml'),
        help='Path to the training config TOML file.',
    )
    add_dataclass_arguments(train_parser, TrainConfig)
    train_parser.set_defaults(command_handler=run_train)

    infer_parser = subparsers.add_parser(
        'infer',
        help='Run model inference.',
        description='Run model inference.',
    )
    infer_parser.add_argument(
        '--config',
        type=Path,
        default=Path('configs/infer.toml'),
        help='Path to the inference config TOML file.',
    )
    infer_parser.add_argument(
        'run_dir',
        nargs='?',
        type=Path,
        help='Run directory that contains best.pt and will receive outputs/.',
    )
    infer_parser.add_argument(
        '--ckpt',
        type=Path,
        default=argparse.SUPPRESS,
        help='Checkpoint path to load manually.',
    )
    infer_parser.add_argument(
        '--input',
        type=Path,
        default=argparse.SUPPRESS,
        help='Input .datalist.csv, image file, or directory of images.',
    )
    infer_parser.add_argument(
        '--test-manifest',
        '--test_manifest',
        dest='test_manifest',
        type=Path,
        default=argparse.SUPPRESS,
        help='Test manifest used for inference/evaluation.',
    )
    infer_parser.add_argument(
        '--output',
        type=Path,
        default=argparse.SUPPRESS,
        help='Output directory. Required when using --ckpt.',
    )
    infer_parser.add_argument(
        '--status',
        nargs='*',
        type=str,
        default=argparse.SUPPRESS,
        help=(
            'Statuses to report during inference. '
            'Pass no values to enable all compatible statuses.'
        ),
    )
    infer_parser.set_defaults(command_handler=run_infer)

    export_parser = subparsers.add_parser(
        'onnx-export',
        help='Export a checkpoint to ONNX.',
        description='Export a checkpoint to ONNX.',
    )
    export_parser.add_argument(
        'run_dir',
        nargs='?',
        type=Path,
        help=(
            'Run directory that contains best.pt and receives '
            'model.<precision>.onnx.'
        ),
    )
    export_parser.add_argument(
        '--ckpt',
        type=Path,
        default=argparse.SUPPRESS,
        help='Checkpoint path to export manually.',
    )
    export_parser.add_argument(
        '--output',
        type=Path,
        default=argparse.SUPPRESS,
        help='Output ONNX file path. Required when using --ckpt.',
    )
    export_parser.add_argument(
        '--precision',
        choices=('fp32', 'fp16', 'bf16', 'int8'),
        default='fp32',
        help='Export precision.',
    )
    export_parser.add_argument(
        '--height',
        type=int,
        default=1080,
        help='Dummy input height used for export.',
    )
    export_parser.add_argument(
        '--width',
        type=int,
        default=1920,
        help='Dummy input width used for export.',
    )
    export_parser.add_argument(
        '--opset',
        type=int,
        default=17,
        help='ONNX opset version.',
    )
    export_parser.set_defaults(command_handler=run_onnx_export)


def run_train(args: argparse.Namespace) -> None:
    _import_train_model()(_load_train_config(args))


def run_infer(args: argparse.Namespace) -> None:
    infer_model = cast(
        Callable[[InferConfig], None],
        import_module('nn_filter.infer').infer_model,
    )
    infer_model(_load_infer_config(args))


def run_onnx_export(args: argparse.Namespace) -> None:
    config = OnnxExportConfig(
        run_dir=getattr(args, 'run_dir', None),
        ckpt=getattr(args, 'ckpt', None),
        output=getattr(args, 'output', None),
        precision=args.precision,
        height=args.height,
        width=args.width,
        opset=args.opset,
    )
    export_onnx = cast(
        Callable[[OnnxExportConfig], Path],
        import_module('nn_filter.onnx_export').export_onnx_model,
    )
    export_onnx(config)


def _load_train_config(args: argparse.Namespace) -> TrainConfig:
    overrides = namespace_overrides(
        args, exclude={'command', 'command_handler', 'config'}
    )
    return load_config(
        TrainConfig,
        config_path=args.config,
        overrides=overrides,
    )


def _load_infer_config(args: argparse.Namespace) -> InferConfig:
    overrides = namespace_overrides(
        args, exclude={'command', 'command_handler', 'config'}
    )
    return load_config(
        InferConfig,
        config_path=args.config,
        overrides=overrides,
    )


def _import_train_model() -> Callable[[TrainConfig], Path]:
    return cast(
        Callable[[TrainConfig], Path],
        import_module('nn_filter.train').train_model,
    )
