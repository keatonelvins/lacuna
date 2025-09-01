"""FSDP2 and DDP distributed training utilities."""

import os
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    fully_shard,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import PreTrainedModel
from loguru import logger

from .config import PretrainConfig, SFTConfig


def init_distributed() -> None:
    """Initialize distributed process group."""
    if not dist.is_available():
        return

    # Check if we're running under torchrun
    if "RANK" not in os.environ:
        return

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    logger.info(f"Initialized distributed: rank {get_rank()}/{get_world_size()}")


def get_rank() -> int:
    """Get current process rank."""
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size() -> int:
    """Get total number of processes."""
    return dist.get_world_size() if dist.is_initialized() else 1


def is_master() -> bool:
    """Check if current process is master (rank 0)."""
    return get_rank() == 0


def get_world_info() -> dict[str, Any]:
    """Get distributed world information."""
    return {
        "rank": get_rank(),
        "world_size": get_world_size(),
        "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
        "is_master": is_master(),
        "distributed": dist.is_initialized(),
    }


def setup_distributed(
    model: PreTrainedModel,
    config: PretrainConfig | SFTConfig,
) -> PreTrainedModel:
    """Setup distributed training based on backend configuration."""

    world_size = get_world_size()

    if world_size == 1:
        logger.info("Single GPU training - no distributed wrapping")
        return model

    if config.dist.backend == "none":
        logger.info("Multi-GPU available but distributed backend disabled")
        return model
    elif config.dist.backend == "ddp":
        return setup_ddp(model, config.model.compile_mode is not None)
    else:  # fsdp
        return setup_fsdp2(model, config.dist.cpu_offload)


def setup_fsdp2(
    model: PreTrainedModel,
    cpu_offload: bool = False,
) -> PreTrainedModel:
    """Setup FSDP2 with per-block wrapping and optimizations."""

    if not dist.is_initialized():
        return model

    logger.info("Setting up FSDP2...")

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        output_dtype=torch.float32,
    )

    # CPU offload if requested
    cpu_offload_policy = CPUOffloadPolicy(pin_memory=True) if cpu_offload else None

    # Apply FSDP2 to transformer blocks with smart resharding
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
        num_layers = len(layers)

        for layer_id, transformer_block in enumerate(layers):
            # Last block optimization: don't reshard since FSDP prefetches
            reshard = layer_id < num_layers - 1

            fully_shard(
                transformer_block,
                mp_policy=mp_policy,
                cpu_offload_policy=cpu_offload_policy,
                reshard_after_forward=reshard,
                sync_module_states=True,
            )

        logger.info(f"Wrapped {num_layers} transformer blocks with FSDP2")

    # Apply root FSDP wrapping (never reshard root)
    fully_shard(
        model,
        mp_policy=mp_policy,
        cpu_offload_policy=cpu_offload_policy,
        reshard_after_forward=False,
        sync_module_states=True,
    )

    logger.info(f"FSDP2 setup complete (cpu_offload={cpu_offload})")
    return model


def setup_ddp(model: PreTrainedModel, is_compiled: bool = False) -> PreTrainedModel:
    """Setup DDP for small model distributed training."""

    if not dist.is_initialized():
        logger.info("DDP disabled - single GPU training")
        return model

    logger.info("Setting up DDP...")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    model = DDP(
        model,
        device_ids=[local_rank],
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
        static_graph=is_compiled,  # Only use static graph if model is compiled
        find_unused_parameters=False,
    )

    logger.info(f"DDP setup complete (static_graph={is_compiled})")
    return model
