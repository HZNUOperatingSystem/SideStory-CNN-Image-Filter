import argparse
import tomllib
import types
from collections.abc import Mapping
from dataclasses import MISSING, Field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, Union, get_args, get_origin

ConfigT = TypeVar('ConfigT')


def add_dataclass_arguments(
    parser: argparse.ArgumentParser, config_type: type[Any]
) -> None:
    for field in fields(config_type):
        if not field.init:
            continue
        option_names = [f'--{field.name.replace("_", "-")}']
        legacy_name = f'--{field.name}'
        if legacy_name not in option_names:
            option_names.append(legacy_name)

        argument_kwargs: dict[str, Any] = {
            'dest': field.name,
            'default': argparse.SUPPRESS,
        }
        if _is_bool_annotation(field.type):
            argument_kwargs['action'] = argparse.BooleanOptionalAction
        else:
            argument_kwargs['type'] = _argument_type(field.type)
        parser.add_argument(*option_names, **argument_kwargs)


def load_config(
    config_type: type[ConfigT],
    *,
    config_path: Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> ConfigT:
    config_values = default_config_values(config_type)

    if config_path is not None:
        file_values = load_toml_config(config_path)
        config_values.update(
            coerce_config_mapping(
                config_type, file_values, base_dir=config_path.parent
            )
        )

    if overrides is not None:
        config_values.update(coerce_config_mapping(config_type, overrides))

    return config_type(**config_values)


def load_toml_config(config_path: Path) -> dict[str, Any]:
    with config_path.open('rb') as config_file:
        raw_data = tomllib.load(config_file)

    if not isinstance(raw_data, dict):
        msg = f'Config file {config_path} must contain a TOML table.'
        raise ValueError(msg)

    return raw_data


def default_config_values(config_type: type[Any]) -> dict[str, Any]:
    if not is_dataclass(config_type):
        msg = f'{config_type!r} is not a dataclass.'
        raise TypeError(msg)

    values: dict[str, Any] = {}
    for field in fields(config_type):
        if not field.init:
            continue
        if field.default is not MISSING:
            values[field.name] = field.default
            continue
        default_factory = field.default_factory
        if default_factory is not MISSING:
            values[field.name] = default_factory()
    return values


def namespace_overrides(
    args: argparse.Namespace, *, exclude: set[str] | None = None
) -> dict[str, Any]:
    excluded = exclude if exclude is not None else set()
    return {
        key: value for key, value in vars(args).items() if key not in excluded
    }


def coerce_config_mapping(
    config_type: type[Any],
    values: Mapping[str, Any],
    *,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    field_map = {
        field.name: field for field in fields(config_type) if field.init
    }
    unknown_keys = sorted(set(values) - set(field_map))
    if unknown_keys:
        joined_keys = ', '.join(unknown_keys)
        msg = f'Unknown config keys for {config_type.__name__}: {joined_keys}'
        raise ValueError(msg)

    return {
        name: _coerce_value(field_map[name], value, base_dir=base_dir)
        for name, value in values.items()
    }


def _argument_type(annotation: Any) -> type[Any]:
    resolved_type = _unwrap_optional(annotation)
    if resolved_type in {str, int, float, Path}:
        return resolved_type
    msg = f'Unsupported CLI argument type: {annotation!r}'
    raise TypeError(msg)


def _coerce_value(
    field: Field[Any], value: Any, *, base_dir: Path | None = None
) -> Any:
    resolved_type = _unwrap_optional(field.type)
    origin = get_origin(resolved_type)

    if resolved_type is Path:
        return _coerce_path(value, base_dir=base_dir)
    if resolved_type in {str, int, float, bool}:
        return resolved_type(value)
    if origin is list:
        item_type = get_args(resolved_type)[0]
        if not isinstance(value, list):
            msg = f'Field {field.name} must be a list.'
            raise TypeError(msg)
        return [
            _coerce_list_item(item_type, item, base_dir=base_dir)
            for item in value
        ]
    msg = f'Unsupported config field type for {field.name}: {field.type!r}'
    raise TypeError(msg)


def _coerce_list_item(
    item_type: Any, value: Any, *, base_dir: Path | None = None
) -> Any:
    if item_type is Path:
        return _coerce_path(value, base_dir=base_dir)
    if item_type in {str, int, float, bool}:
        return item_type(value)
    msg = f'Unsupported list item type: {item_type!r}'
    raise TypeError(msg)


def _coerce_path(value: Any, *, base_dir: Path | None = None) -> Path:
    path = value if isinstance(value, Path) else Path(value)
    if base_dir is not None and not path.is_absolute():
        return base_dir / path
    return path


def _unwrap_optional(annotation: Any) -> Any:
    origin = get_origin(annotation)
    if origin not in {types.UnionType, Union}:
        return annotation

    args = tuple(arg for arg in get_args(annotation) if arg is not type(None))
    if len(args) == 1:
        return args[0]
    return annotation


def _is_bool_annotation(annotation: Any) -> bool:
    return _unwrap_optional(annotation) is bool
