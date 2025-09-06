"""Model setup and optimization utilities."""

import os
import functools
from contextlib import redirect_stdout, redirect_stderr

import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from torch.nn.attention import sdpa_kernel, SDPBackend
from transformers import PreTrainedModel, AutoModelForCausalLM
from kernels import kernelize
from cut_cross_entropy.transformers import cce_patch
from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance
from loguru import logger

from .config import (
    ActivationCheckpointConfig,
    ModelConfig,
    LacunaConfig,
)

ATTN_IMPL_MAP = {
    "EAGER": "eager",
    "FA3": "kernels-community/flash-attn3",
    "SDPA": "sdpa",
}


def setup_model(config: LacunaConfig) -> PreTrainedModel:
    """Load and fully configure model for training."""
    model_path = config.model.name

    if config.checkpoint.resume_path and (config.checkpoint.resume_path / "config.json").exists():
        model_path = config.checkpoint.resume_path

    logger.info(f"Loading model: {model_path} with {config.model.attention}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation=ATTN_IMPL_MAP[config.model.attention],
        use_cache=False,
    )

    model = apply_liger_patches(model, config.model)
    model = apply_cut_cross_entropy(model, config.model)
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
    ):  # silence unaesthetic liger print (lol liger print)
        _apply_liger_kernel_to_instance(
            model,
            fused_linear_cross_entropy=not config.cce,  # avoid double patching
        )

    return model


def apply_cut_cross_entropy(model: PreTrainedModel, config: ModelConfig) -> PreTrainedModel:
    """Apply Cut Cross Entropy optimization to model."""
    if not config.cce:
        return model

    logger.info("Applying Cut Cross Entropy")
    model = cce_patch(
        model,
        accum_e_fp32=config.accum_fp32,
        accum_c_fp32=config.accum_fp32,
    )

    return model


def apply_activation_checkpointing(model: PreTrainedModel, ac_config: ActivationCheckpointConfig) -> PreTrainedModel:
    """Apply activation checkpointing to transformer blocks."""
    if ac_config.mode == "none":
        return model

    layers = model.model.layers

    if ac_config.mode == "full":
        checkpoint_freq = 1
    elif ac_config.mode == "partial":
        checkpoint_freq = ac_config.stride

    for idx, layer in enumerate(layers):
        if idx % checkpoint_freq == 0:
            layers[idx] = checkpoint_wrapper(layer, preserve_rng_state=False)

    logger.info(f"Applied {ac_config.mode} activation checkpointing")
    return model


def apply_torch_compile(model: PreTrainedModel, config: LacunaConfig) -> PreTrainedModel:
    """Apply torch.compile to each individual transformer block."""
    # Patch model.forward with SDPA kernel if using SDPA
    if config.model.attention == "SDPA":
        backends = [SDPBackend.FLASH_ATTENTION]
        capability = torch.cuda.get_device_capability()
        if capability[0] >= 9:  # H100 is 9.0, H200/B200 are 9.0+
            backends.insert(0, SDPBackend.CUDNN_ATTENTION)

        original_forward = model.forward

        @functools.wraps(original_forward)
        def sdpa_forward(*args, **kwargs):
            with sdpa_kernel(backends, set_priority=True):
                return original_forward(*args, **kwargs)

        model.forward = sdpa_forward
        backend_names = [b.name for b in backends]
        logger.info(f"Patched model.forward with SDPA backends: {backend_names}")

    if not config.model.compile_mode:
        return model

    fullgraph = config.model.attention == "SDPA"

    torch._dynamo.config.cache_size_limit = 256
    torch._dynamo.config.suppress_errors = True

    layers = model.model.layers
    for idx, layer in enumerate(layers):
        compiled_layer = torch.compile(
            layer,
            fullgraph=fullgraph,
            mode=config.model.compile_mode,
        )
        layers[idx] = compiled_layer
    logger.info(f"Applied torch.compile (mode={config.model.compile_mode}, fullgraph={fullgraph})")

    return model


def apply_kernelize(model: PreTrainedModel, config: ModelConfig) -> PreTrainedModel:
    """Apply HuggingFace kernels.kernelize to the model."""
    if not config.kernelize:
        return model

    model = kernelize(model)

    return model
