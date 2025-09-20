"""Model setup and optimization utilities."""

import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from transformers import PreTrainedModel, AutoModelForCausalLM
from kernels import kernelize, Mode
from fla.modules.fused_linear_cross_entropy import FusedLinearCrossEntropyLoss
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

    if config.checkpoint.resume_from:
        model_path = config.checkpoint.resume_from

    logger.info(f"Loading model: {model_path} with {config.model.attention}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation=ATTN_IMPL_MAP[config.model.attention],
    )
    model.config.use_cache = False  # needed for ac to work

    # TODO: reference flame for order of patches
    model.criterion = FusedLinearCrossEntropyLoss()
    model = apply_kernelize(model, config.model)
    model = apply_activation_checkpointing(model, config.ac)
    model = apply_torch_compile(model, config)

    model.train()

    return model


def apply_activation_checkpointing(model: PreTrainedModel, ac_config: ActivationCheckpointConfig) -> PreTrainedModel:
    """Apply activation checkpointing if enabled."""
    if ac_config.stride == 0:
        return model

    # TODO: may need SAC for MoE's
    for idx, layer in enumerate(model.model.layers):
        if idx % ac_config.stride == 0:
            model.model.layers[idx] = checkpoint_wrapper(layer)

    return model


def apply_torch_compile(model: PreTrainedModel, config: LacunaConfig) -> PreTrainedModel:
    """Apply torch.compile if enabled."""
    if not config.model.compile_mode:
        return model

    torch._dynamo.config.cache_size_limit = 256
    torch._dynamo.config.suppress_errors = True

    for layer in model.model.layers:
        layer.compile(mode=config.model.compile_mode)

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
