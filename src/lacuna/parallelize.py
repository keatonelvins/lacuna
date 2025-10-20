"""Distributed parallelization strategies for training."""

import torch
import torch.nn as nn
from typing import Any
from loguru import logger
from contextlib import nullcontext
from torch.distributed import DeviceMesh
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._composable.fsdp import (
    MixedPrecisionPolicy,
    fully_shard,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)

from lacuna.config import TrainConfig, DistributedConfig, CompileConfig
from lacuna.utils import get_world_size


def apply_parallelisms(model: nn.Module, config: TrainConfig) -> tuple[nn.Module, DeviceMesh | None, Any]:
    """Apply parallelisms: TP → AC → Compile → FSDP/HSDP (order important!!!)"""
    mesh = build_mesh(config)
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if mesh else nullcontext()

    # TODO: apply_tp(model, config.dist)
    apply_activation_checkpointing(model, config.ac.stride)
    apply_compile(model, config.compile)
    apply_fsdp(model, mesh, config.dist)

    return model, mesh, amp


def build_mesh(config: TrainConfig) -> DeviceMesh | None:
    """Create device mesh for FSDP or HSDP."""
    world_size = get_world_size()
    if world_size == 1:
        return None

    dp_replicate, dp_shard = config.dist.dp_replicate, config.dist.dp_shard

    assert dp_replicate * dp_shard == world_size, (
        f"dp_replicate ({dp_replicate}) * dp_shard ({dp_shard}) != world_size ({world_size})"
    )

    if dp_replicate > 1 and dp_shard > 1:  # HSDP: 2D mesh
        mesh = init_device_mesh("cuda", [dp_replicate, dp_shard], mesh_dim_names=["dp_replicate", "dp_shard"])
        mesh["dp_replicate", "dp_shard"]._flatten(mesh_dim_name="dp")
        return mesh
    elif dp_shard > 1:  # FSDP: 1D mesh
        return init_device_mesh("cuda", [dp_shard], mesh_dim_names=["dp_shard"])
    else:
        raise ValueError(f"Invalid DP config: replicate={dp_replicate}, shard={dp_shard}")


def apply_activation_checkpointing(model: nn.Module, stride: int) -> None:
    """Apply activation checkpointing to model layers."""
    if stride == 0:
        return

    for idx, layer in enumerate(model.model.layers):
        if idx % stride == 0:
            model.model.layers[idx] = checkpoint_wrapper(layer, preserve_rng_state=False)

    logger.info(f"Applied activation checkpointing (stride={stride})")


def apply_compile(model: nn.Module, compile_config: CompileConfig) -> None:
    """Apply torch.compile to model layers."""
    if compile_config.mode is None:
        return

    torch._dynamo.config.cache_size_limit = 256
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.capture_scalar_outputs = True

    for layer in model.model.layers:
        layer.compile(
            mode=compile_config.mode,
            fullgraph=compile_config.fullgraph,
            dynamic=compile_config.dynamic,
        )

    logger.info(f"Applied torch.compile (mode={compile_config.mode})")


def apply_fsdp(model: nn.Module, mesh: DeviceMesh | None, config: DistributedConfig) -> None:
    """Apply FSDP or HSDP to model."""
    if mesh is None:
        return

    fsdp_kwargs = dict(
        mesh=mesh, mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    )

    for block in model.model.layers:
        fully_shard(block, reshard_after_forward=config.reshard, **fsdp_kwargs)

    if not model.config.tie_word_embeddings:
        fully_shard(model.model.embed_tokens, reshard_after_forward=config.reshard, **fsdp_kwargs)
        fully_shard(model.model.norm, reshard_after_forward=False, **fsdp_kwargs)

    fully_shard(model, reshard_after_forward=False, **fsdp_kwargs)

    logger.info(f"Model sharding complete (reshard_after_forward={config.reshard})")
    return model
