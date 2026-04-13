import argparse
from pathlib import Path

from nn_filter.train import TrainConfig, train_model

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


def run_train(args: argparse.Namespace) -> None:
    overrides = namespace_overrides(
        args, exclude={'command', 'command_handler', 'config'}
    )
    config = load_config(
        TrainConfig, config_path=args.config, overrides=overrides
    )
    train_model(config)
