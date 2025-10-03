"""Distributed training setup and utils."""

import os
import time
import torch
import random
import contextlib
import numpy as np
import torch.distributed as dist
from transformers import PreTrainedModel
from torch.distributed.device_mesh import init_device_mesh, DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard
from loguru import logger

from lacuna.config import LacunaConfig


def init_dist(config: LacunaConfig) -> None:
    """Initialize distributed process group and return world size."""
    if "LOCAL_RANK" not in os.environ:
        set_seed(config.trainer.seed)
        return

    backend = "cuda:nccl,cpu:gloo" if config.dist.cpu_offload else "nccl"
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    dist.init_process_group(backend=backend, device_id=local_rank)
    set_seed(config.trainer.seed)


def destroy_dist() -> None:
    if dist.is_initialized():
        if is_master():
            time.sleep(2)  # give other ranks a second to finish
        dist.destroy_process_group()


def get_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def is_master() -> bool:
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


def get_dp_params(config: LacunaConfig) -> tuple[int, int]:
    world_size = get_world_size()
    if world_size == 1:
        return 1, 1

    rep = config.dist.dp_replicate or config.torchrun.nnodes
    shard = config.dist.dp_shard or config.torchrun.nproc_per_node

    if rep * shard != world_size:
        raise ValueError(f"dp_replicate x dp_shard={rep} x {shard} != world_size={world_size}")

    return rep, shard


def get_dp_mesh(config: LacunaConfig) -> DeviceMesh | None:
    world_size = get_world_size()
    rep, shard = get_dp_params(config)

    if rep > 1 and shard > 1:
        mesh = init_device_mesh("cuda", [rep, shard], mesh_dim_names=["dp_replicate", "dp_shard"])
    elif shard > 1:
        mesh = init_device_mesh("cuda", [shard], mesh_dim_names=["dp_shard"])
    elif rep > 1:
        raise ValueError("Invalid config: DDP unsupported, please use FSDP instead")
    else:
        raise ValueError(f"Invalid config: dp_replicate=1, dp_shard=1 but world_size={world_size}")

    logger.info(f"Mesh setup complete (replicate={rep}, shard={shard}, world={world_size})")
    return mesh


def setup_dist(model: PreTrainedModel, config: LacunaConfig) -> tuple[PreTrainedModel, torch.autocast, DeviceMesh | None]:
    """Returns model and autocast context manager (null for FSDP as handled internally)."""
    if get_world_size() == 1:
        return model.cuda(), torch.autocast("cuda", dtype=torch.bfloat16), None

    mesh = get_dp_mesh(config)

    return setup_fsdp2(model, config, mesh), contextlib.nullcontext(), mesh


def setup_fsdp2(model, config, mesh) -> PreTrainedModel:
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    offload = CPUOffloadPolicy() if config.dist.cpu_offload else None
    reshard = config.dist.reshard_after_forward

    for block in model.model.layers:
        fully_shard(block, mesh=mesh, mp_policy=mp_policy, offload_policy=offload, reshard_after_forward=reshard)

    if not model.config.tie_word_embeddings:
        fully_shard(model.model.embed_tokens, mesh=mesh, mp_policy=mp_policy, reshard_after_forward=reshard)
        if config.model.backend != "hf": # liger/lacuna use flce and don't run lm_head directly
            fully_shard(model.model.norm, mesh=mesh, mp_policy=mp_policy, reshard_after_forward=False)
        else:
            fully_shard([model.lm_head, model.model.norm], mesh=mesh, mp_policy=mp_policy, reshard_after_forward=False)

    fully_shard(model, mesh=mesh, mp_policy=mp_policy, offload_policy=offload, reshard_after_forward=False)

    logger.info(f"Model sharding complete (cpu_offload={config.dist.cpu_offload}, reshard_after_forward={reshard})")
    return model
