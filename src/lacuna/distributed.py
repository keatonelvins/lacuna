"""FSDP2 and DDP distributed training utilities."""

import os
import random

import numpy as np
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


def destroy_distributed() -> None:
    """Destroy distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


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


def set_seed(seed: int) -> int:
    """Set seeds for reproducibility across all RNGs."""
    if dist.is_initialized():
        seed_tensor = torch.tensor(seed, dtype=torch.long, device="cuda")
        dist.broadcast(seed_tensor, src=0)
        seed = int(seed_tensor.item())

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    return seed


def get_dp_mesh(config: LacunaConfig) -> DeviceMesh | None:
    world_size = get_world_size()
    if world_size == 1:
        return None

    rep = config.dist.dp_replicate or config.torchrun.nnodes
    shard = config.dist.dp_shard or config.torchrun.nproc_per_node

    if rep * shard != world_size:
        raise ValueError(f"dp_replicate×dp_shard={rep}×{shard} ≠ world_size={world_size}")

    if rep > 1 and shard > 1:
        mode, mesh = "HSDP", init_device_mesh("cuda", [rep, shard], mesh_dim_names=["dp_replicate", "dp_shard"])
    elif rep > 1:
        mode, mesh = "DDP", None
    elif shard > 1:
        mode, mesh = "FSDP", init_device_mesh("cuda", [shard], mesh_dim_names=["dp_shard"])
    else:
        raise ValueError(f"Invalid config: dp_replicate=1, dp_shard=1 but world_size={world_size}")

    logger.info(f"{mode} mesh: replicate={rep}, shard={shard}, world={world_size}")
    return mesh


def setup_distributed(model: PreTrainedModel, config: LacunaConfig) -> PreTrainedModel:
    world_size = get_world_size()
    if world_size == 1:
        return model

    mesh = get_dp_mesh(config)

    if mesh:
        return setup_fsdp2(model, config, mesh)
    else:
        return setup_ddp(model, config)


def setup_fsdp2(model, config, mesh) -> PreTrainedModel:
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

    logger.info(f"Model sharding complete (cpu_offload={config.dist.cpu_offload})")
    return model


def setup_ddp(model: PreTrainedModel, config: LacunaConfig) -> PreTrainedModel:
    # TODO: document flags and values
    scale = (12 * model.config.hidden_size**2) / 1e8
    bucket = 25 * (1 + scale)
    bucket *= 1.5 if get_world_size() > 32 else 1
    clipped_cap = int(min(max(bucket, 10), 250))
    is_compiled = config.model.compile_mode is not None
    model = DDP(
        model,
        device_ids=[get_rank()],
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
        static_graph=is_compiled,  # only use static graph if model is compiled
        find_unused_parameters=False,
        bucket_cap_mb=clipped_cap,
    )

    logger.info(f"DDP setup complete (static_graph={is_compiled})")
    return model
