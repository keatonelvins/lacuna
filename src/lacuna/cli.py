"""Entry point with TOML config loading and CLI overrides."""

import sys
from pathlib import Path
from typing import Type, TypeVar

import tomllib
from pydantic_settings import BaseSettings

from lacuna.config import PretrainConfig, SFTConfig
from lacuna.trainer import train

T = TypeVar("T", bound=BaseSettings)


def parse_argv(config_cls: Type[T], args: list[str] | None = None) -> T:
    """Parse TOML config file and CLI overrides into pydantic settings"""
    if args is None:
        args = sys.argv[1:]

    # First arg is TOML file path if it exists and doesn't start with --
    if args and not args[0].startswith("--"):
        config_path = Path(args[0])
        cli_args = args[1:]

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "rb") as f:
            toml_data = tomllib.load(f)
    else:
        toml_data = {}
        cli_args = args

    # Create config with TOML data as defaults, then apply CLI overrides
    return config_cls(**toml_data, _cli_parse_args=cli_args)


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
