"""FSDP setup and distributed training utilities."""

import os
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    ShardingStrategy,
    fully_shard,
)
from transformers import PreTrainedModel
from loguru import logger

from .config import FSDPShardingStrategy


def init_distributed() -> None:
    """Initialize distributed process group."""
    if not dist.is_available():
        return

    # Check if we're running under torchrun
    if "RANK" in os.environ:
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


def setup_fsdp(
    model: PreTrainedModel,
    reshard_after_forward: bool = True,
    cpu_offload: bool = False,
    sharding_strategy: FSDPShardingStrategy = FSDPShardingStrategy.FULL_SHARD,
) -> PreTrainedModel:
    """Setup FSDP wrapping."""

    if not dist.is_initialized():
        logger.info("FSDP disabled - single GPU training")
        return model

    logger.info("Setting up FSDP...")

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        output_dtype=torch.bfloat16,
    )

    cpu_offload_policy = CPUOffloadPolicy(pin_memory=True) if cpu_offload else None

    torch_strategy = getattr(ShardingStrategy, sharding_strategy.value)

    logger.info(
        f"FSDP Config: strategy={sharding_strategy.value}, cpu_offload={cpu_offload}"
    )

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
        num_layers = len(layers)

        for layer_id, transformer_block in enumerate(layers):
            # Reshard all but last layer to save memory
            layer_reshard = reshard_after_forward and (layer_id < num_layers - 1)
            fully_shard(
                transformer_block,
                mp_policy=mp_policy,
                cpu_offload_policy=cpu_offload_policy,
                reshard_after_forward=layer_reshard,
                sharding_strategy=torch_strategy,
                sync_module_states=True,  # Ensure consistent initialization
            )

        logger.info(f"Wrapped {num_layers} transformer layers with FSDP")

    # Wrap entire model with FSDP
    fully_shard(
        model,
        mp_policy=mp_policy,
        cpu_offload_policy=cpu_offload_policy,
        reshard_after_forward=reshard_after_forward,
        sharding_strategy=torch_strategy,
        sync_module_states=True,
    )

    logger.info("FSDP setup complete")
    return model
