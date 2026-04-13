import argparse
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import cast

from nn_filter.config import InferConfig, TrainConfig

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
        required=True,
        help='Input .datalist.csv, image file, or directory of images.',
    )
    infer_parser.add_argument(
        '--output',
        type=Path,
        default=argparse.SUPPRESS,
        help='Output directory. Required when using --ckpt.',
    )
    infer_parser.set_defaults(command_handler=run_infer)


def run_train(args: argparse.Namespace) -> None:
    overrides = namespace_overrides(
        args, exclude={'command', 'command_handler', 'config'}
    )
    config = load_config(
        TrainConfig, config_path=args.config, overrides=overrides
    )
    train_model = cast(
        Callable[[TrainConfig], None],
        import_module('nn_filter.train').train_model,
    )
    train_model(config)


def run_infer(args: argparse.Namespace) -> None:
    config = InferConfig(
        run_dir=getattr(args, 'run_dir', None),
        ckpt=getattr(args, 'ckpt', None),
        input=args.input,
        output=getattr(args, 'output', None),
    )
    infer_model = cast(
        Callable[[InferConfig], None],
        import_module('nn_filter.infer').infer_model,
    )
    infer_model(config)
