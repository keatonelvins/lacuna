"""Model setup and optimization utils."""

import torch
from loguru import logger
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from kernels import kernelize, Mode
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoConfig
from liger_kernel.transformers.auto_model import AutoLigerKernelForCausalLM
from lacuna.moe import AutoLacunaModelForCausalLM

from lacuna.config import (
    ActivationCheckpointConfig,
    ModelConfig,
    LacunaConfig,
)


def setup_model(config: LacunaConfig) -> PreTrainedModel:
    """Load and fully configure model for training."""
    logger.info(f"Loading model: {config.model.name} with {config.model.attention}")

    if config.model.backend == "lacuna":
        model_factory = AutoLacunaModelForCausalLM
    elif config.model.backend == "liger":
        model_factory = AutoLigerKernelForCausalLM
    else:
        model_factory = AutoModelForCausalLM

    if config.checkpoint.resume_from:  # dcp will load weights later
        if config.model.backend == "lacuna":
            raise ValueError("Cannot resume from checkpoint for Lacuna model")

        model_config = AutoConfig.from_pretrained(
            config.model.name,
            attn_implementation=config.model.attention,
        )
        model_config.use_cache = False
        model = model_factory.from_config(model_config, dtype=torch.bfloat16)
        logger.info(f"Initialized model structure for checkpoint resume from {config.checkpoint.resume_from}")
    else:
        model = model_factory.from_pretrained(
            config.model.name,
            dtype=torch.bfloat16,
            attn_implementation=config.model.attention,
        )
        model.config.use_cache = False

    model = apply_kernelize(model, config.model)
    model = apply_activation_checkpointing(model, config.ac)
    model = apply_torch_compile(model, config.model)

    model.train()

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


def apply_activation_checkpointing(model: PreTrainedModel, ac_config: ActivationCheckpointConfig) -> PreTrainedModel:
    """Apply activation checkpointing if enabled."""
    if ac_config.stride == 0:
        return model

    for idx, layer in enumerate(model.model.layers):
        if idx % ac_config.stride == 0:
            model.model.layers[idx] = checkpoint_wrapper(layer, preserve_rng_state=False)

    return model


def apply_torch_compile(model: PreTrainedModel, config: ModelConfig) -> PreTrainedModel:
    """Apply torch.compile if enabled."""
    if not config.compile_mode:
        return model

    torch._dynamo.config.cache_size_limit = 256
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.capture_scalar_outputs = True

    for layer in model.model.layers:
        layer.compile(mode=config.compile_mode)

    return model
