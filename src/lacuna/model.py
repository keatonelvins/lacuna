"""Model loading and initialization."""

from types import SimpleNamespace

import torch
from loguru import logger
from transformers import AutoConfig, AutoModelForCausalLM
from liger_kernel.transformers.monkey_patch import _apply_liger_kernel as apply_liger
from torchtitan.config.job_config import Float8Linear
from torchtitan.components.quantization.float8 import Float8LinearConverter

from lacuna.config import TrainConfig
from lacuna.moe import apply_tt_moe


def build_model(config: TrainConfig) -> tuple[torch.nn.Module, object | None]:
    """Build model on meta device (no memory allocation). Returns (model, fp8_converter)."""
    logger.info(f"Building model: {config.model.name}")
    model_config = AutoConfig.from_pretrained(config.model.name, attn_implementation=config.model.attn)
    model_config.use_cache = False  # necessary for ac

    with torch.device("meta"):
        if config.model.lacuna:
            apply_liger(model_config.model_type)
            apply_tt_moe(model_config.model_type)
        model = AutoModelForCausalLM.from_config(model_config)

    logger.info("Uninitialized model built on meta device!")
    fp8_converter = apply_fp8(model, config)

    return model, fp8_converter


def initialize_buffers(model: torch.nn.Module) -> None:
    """Initialize buffers after to_empty() since they contain uninitialized memory."""
    seen_buffers = {name: False for name, _ in model.named_buffers()}

    for name, buf in model.named_buffers():
        if "mlp.tokens_per_expert" in name or "mlp.expert_bias" in name:
            buf.zero_()
            seen_buffers[name] = True
        elif "model.rotary_emb.inv_freq" == name:
            logger.info(f"Initializing rotary_emb.inv_freq for {name}")
            model_emb = model.model.rotary_emb
            inv_freq, model_emb.attention_scaling = model_emb.rope_init_fn(model_emb.config, model_emb.inv_freq.device)
            model_emb.inv_freq.copy_(inv_freq)
            seen_buffers[name] = True

    # make sure we handled all buffers that may need initialization
    assert all(seen_buffers.values()), f"Unknown buffers detected that may need initialization: {seen_buffers}"


def apply_fp8(model: torch.nn.Module, config: TrainConfig):
    """Apply FP8 quantization (tensorwise with FSDP optimization). Returns converter for post-optimizer hook."""
    if not config.model.fp8:
        return None

    logger.info("NOTE: FP8 is untested, may need ao nightly or significant tuning")

    # spoof the torchtitan job config
    enable_fsdp = config.dist.dp_shard > 1
    fp8_config = Float8Linear(
        enable_fsdp_float8_all_gather=enable_fsdp, precompute_float8_dynamic_scale_for_fsdp=enable_fsdp
    )
    job_config = SimpleNamespace(
        quantize=SimpleNamespace(linear=SimpleNamespace(float8=fp8_config)),
        compile=SimpleNamespace(enable=True, components=["model"]),
    )
    parallel_dims = SimpleNamespace(dp_shard_enabled=enable_fsdp)

    converter = Float8LinearConverter(job_config, parallel_dims)
    converter.convert(model)
    return converter
