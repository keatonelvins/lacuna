"""Model setup and optimization utilities."""

import os
from contextlib import redirect_stdout, redirect_stderr

import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from transformers import PreTrainedModel, AutoModelForCausalLM
from kernels import kernelize, Mode
from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance
from loguru import logger

from .config import (
    ActivationCheckpointConfig,
    ModelConfig,
    LacunaConfig,
)

ATTN_IMPL_MAP = {
    "FA3": "kernels-community/flash-attn3",
}


def setup_model(config: LacunaConfig) -> PreTrainedModel:
    """Load and fully configure model for training."""
    model_path = config.model.name

    if config.checkpoint.resume_from and (config.checkpoint.resume_from / "config.json").exists():
        model_path = config.checkpoint.resume_from

    logger.info(f"Loading model: {model_path} with {config.model.attention}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation=ATTN_IMPL_MAP[config.model.attention],
    )

    model = apply_liger_patches(model, config.model)
    model = apply_kernelize(model, config.model)
    model = apply_activation_checkpointing(model, config.ac)
    model = apply_torch_compile(model, config)

    model = model.cuda()
    model.train()

    return model


def apply_liger_patches(model: PreTrainedModel, config: ModelConfig) -> PreTrainedModel:
    """Apply Liger kernel patches if enabled"""
    if not config.liger:
        return model

    with (
        open(os.devnull, "w") as devnull,
        redirect_stdout(devnull),
        redirect_stderr(devnull),
    ):  # silence ugly liger print (lol liger print)
        _apply_liger_kernel_to_instance(model)

    return model


def apply_activation_checkpointing(model: PreTrainedModel, ac_config: ActivationCheckpointConfig) -> PreTrainedModel:
    """Apply activation checkpointing if enabled."""
    if ac_config.mode == "none":
        return model

    layers = model.model.layers

    if ac_config.mode == "full":
        checkpoint_freq = 1
    elif ac_config.mode == "partial":
        checkpoint_freq = ac_config.stride

    # TODO: torch.compile has it's own AC (not stable as of 2.8.0)
    for idx, layer in enumerate(layers):
        if idx % checkpoint_freq == 0:
            layers[idx] = checkpoint_wrapper(layer, preserve_rng_state=False)

    return model


def apply_torch_compile(model: PreTrainedModel, config: LacunaConfig) -> PreTrainedModel:
    """Apply torch.compile if enabled."""
    if not config.model.compile_mode:
        return model

    torch._dynamo.config.cache_size_limit = 256
    torch._dynamo.config.suppress_errors = True

    layers = model.model.layers
    for idx, layer in enumerate(layers):
        compiled_layer = torch.compile(
            layer,
            fullgraph=False,
            mode=config.model.compile_mode,
        )
        layers[idx] = compiled_layer

    return model


def apply_kernelize(model: PreTrainedModel, config: ModelConfig) -> PreTrainedModel:
    """Apply HuggingFace kernels.kernelize if enabled."""
    if not config.kernelize:
        return model

    mode = Mode.TRAINING
    if config.compile_mode:
        mode |= Mode.TORCH_COMPILE

    model = kernelize(model, mode=mode)

    return model
