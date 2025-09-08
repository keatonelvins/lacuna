"""FSDP2 and DDP distributed training utilities."""

import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh, DeviceMesh
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    fully_shard,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import PreTrainedModel
from loguru import logger

from .config import LacunaConfig


def init_distributed() -> None:
    """Initialize distributed process group."""
    if not dist.is_available():
        return

    if "RANK" not in os.environ:
        return

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    logger.info(f"Initialized distributed: rank {get_rank()}/{get_world_size()}")


def get_local_rank() -> int:
    """Get current process local rank."""
    return torch.cuda.current_device()


def get_rank() -> int:
    """Get current process rank."""
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size() -> int:
    """Get total number of processes."""
    return dist.get_world_size() if dist.is_initialized() else 1


def is_master() -> bool:
    """Check if current process is master (rank 0)."""
    return get_rank() == 0


def get_hsdp_mesh(config: LacunaConfig) -> DeviceMesh:
    """Create 2D device mesh for HSDP."""
    world_size = get_world_size()

    if not config.dist.hsdp:
        return None
    if world_size == 1:
        logger.warning("HSDP requested but world_size=1, using standard FSDP")
        return None
    
    if config.torchrun.nnodes > 1:
        dp_replicate = config.torchrun.nnodes
        dp_shard = world_size // config.torchrun.nnodes   
    else:
        dp_replicate, dp_shard = 2, world_size // 2
    
    logger.info(f"HSDP mesh: {dp_replicate}×{dp_shard} = {world_size} GPUs")
    mesh = init_device_mesh("cuda", [dp_replicate, dp_shard], mesh_dim_names=["dp_replicate", "dp_shard"])
    return mesh


def setup_distributed(model: PreTrainedModel, config: LacunaConfig) -> PreTrainedModel:
    """Setup distributed training based on backend configuration."""

    world_size = get_world_size()

    if world_size == 1:
        logger.info("Single GPU training - no distributed wrapping")
        return model
    elif config.dist.backend == "DDP":
        return setup_ddp(model, config)
    else:  # fsdp
        return setup_fsdp2(model, config)


def setup_fsdp2(model: PreTrainedModel, config: LacunaConfig) -> PreTrainedModel:
    """Setup FSDP2 with per-block wrapping and optimizations."""

    if not dist.is_initialized():
        return model

    mesh = get_hsdp_mesh(config)
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    cpu_offload_policy = CPUOffloadPolicy(pin_memory=True) if config.dist.cpu_offload else None

    for i, block in enumerate(model.model.layers):
        # Last block: don't reshard since FSDP prefetches
        reshard = i < len(model.model.layers) - 1

        fully_shard(
            block,
            mesh=mesh,
            mp_policy=mp_policy,
            offload_policy=cpu_offload_policy,
            reshard_after_forward=reshard,
        )

    model = fully_shard(
        model,
        mesh=mesh,
        mp_policy=mp_policy,
        offload_policy=cpu_offload_policy,
        reshard_after_forward=False,
    )

    logger.info(f"{'HSDP' if config.dist.hsdp else 'FSDP2'} setup complete (cpu_offload={config.dist.cpu_offload})")
    return model


def setup_ddp(model: PreTrainedModel, config: LacunaConfig) -> PreTrainedModel:
    """Setup DDP for small model distributed training."""

    if not dist.is_initialized():
        logger.info("DDP disabled - single GPU training")
        return model

    logger.info("Setting up DDP...")

    # TODO: tune bucket_cap_mb and document these settings
    is_compiled = config.model.compile_mode is not None
    model = DDP(
        model,
        device_ids=[get_rank()],
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
        static_graph=is_compiled,  # only use static graph if model is compiled
        find_unused_parameters=False,
        bucket_cap_mb=100,
    )

    logger.info(f"DDP setup complete (static_graph={is_compiled})")
    return model
