"""Entry point with TOML config loading and CLI overrides."""

import sys
from pathlib import Path
from typing import Type, TypeVar

import tomllib
from pydantic_settings import BaseSettings

from lacuna.config import PretrainConfig, SFTConfig
from lacuna.trainer import train

T = TypeVar("T", bound=BaseSettings)


def parse_config_file(config_cls: Type[T], config_path: Path) -> T:
    """Parse TOML config file into pydantic settings."""
    with open(config_path, "rb") as f:
        toml_data = tomllib.load(f)
    return config_cls(**toml_data)


def parse_argv(config_cls: Type[T]) -> T:
    """Parse CLI args with @ syntax for TOML files."""
    args = sys.argv[1:]

    if len(args) >= 2 and args[0] == "@":
        config_path = Path(args[1])
        cli_overrides = args[2:]
    elif len(args) >= 1 and args[0].startswith("@"):
        config_path = Path(args[0][1:])
        cli_overrides = args[1:]
    else:
        config_path = None
        cli_overrides = args

    if config_path:
        config = parse_config_file(config_cls, config_path)
    else:
        config = config_cls()

    # Apply CLI overrides
    if cli_overrides:
        override_dict = {}
        i = 0
        while i < len(cli_overrides):
            arg = cli_overrides[i]
            if arg.startswith("--"):
                key = arg[2:].replace("-", "_")
                if i + 1 < len(cli_overrides) and not cli_overrides[i + 1].startswith(
                    "--"
                ):
                    value = cli_overrides[i + 1]
                    # Try to parse as int/float/bool
                    if value.lower() in ("true", "false"):
                        override_dict[key] = value.lower() == "true"
                    elif value.isdigit():
                        override_dict[key] = int(value)
                    else:
                        try:
                            override_dict[key] = float(value)
                        except ValueError:
                            override_dict[key] = value
                    i += 2
                else:
                    override_dict[key] = True
                    i += 1
            else:
                i += 1

        # Create new instance with overrides
        config_dict = config.model_dump()
        config_dict.update(override_dict)
        config = config_cls(**config_dict)

    return config


def pretrain_main():
    """Entry point for pretraining."""
    config = parse_argv(PretrainConfig)
    print(f"Starting pretraining with model: {config.model.name}")
    train(config)


def sft_main():
    """Entry point for SFT."""
    config = parse_argv(SFTConfig)
    print(f"Starting SFT with model: {config.model.name}")
    train(config)
