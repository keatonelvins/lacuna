"""Optimizer setup."""

import torch
from torch import nn
from dion import Muon
from torchao.optim import AdamW8bit
from torch.optim import Optimizer, AdamW
from transformers import PreTrainedModel
from torch.distributed import DeviceMesh
from transformers.trainer_pt_utils import get_parameter_names

from lacuna.config import TrainConfig


# ref: https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py
def get_decay_parameter_names(model) -> list[str]:
    exclude_params = [r"bias", r"layernorm", r"rmsnorm", r"(?:^|\.)norm(?:$|\.)", r"_norm(?:$|\.)"]
    exclude_params.extend([r"embed_tokens", r"wte", r"wpe", r"embeddings"])  # also exclude embeddings!
    decay_parameters = get_parameter_names(model, [nn.LayerNorm], exclude_params)
    return decay_parameters


# ref: https://github.com/PrimeIntellect-ai/prime-rl/blob/main/src/prime_rl/trainer/optim.py
def use_muon(name: str, param: torch.Tensor) -> bool:
    if param.ndim < 2:
        return False
    if "lm_head" in name:
        return False
    if "embed_tokens" in name:
        return False
    return True


def get_optimizer_params(model, config: TrainConfig) -> list[dict]:
    """Build optimizer param groups: decay/no_decay or muon/adamw"""
    if config.optim.name == "muon":
        muon_params = [p for n, p in model.named_parameters() if p.requires_grad and use_muon(n, p)]
        adamw_params = [p for n, p in model.named_parameters() if p.requires_grad and not use_muon(n, p)]
        optimizer_params = [
            {"params": muon_params, "algorithm": "muon", "adjust_lr": "rms_norm"},
            {"params": adamw_params, "algorithm": "adamw"},
        ]
    else:
        decay_names = get_decay_parameter_names(model)
        decay_params = [p for n, p in model.named_parameters() if p.requires_grad and n in decay_names]
        no_decay_params = [p for n, p in model.named_parameters() if p.requires_grad and n not in decay_names]
        optimizer_params = [
            {"params": decay_params, "weight_decay": config.optim.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

    return optimizer_params


def build_optimizer(model: PreTrainedModel, config: TrainConfig, mesh: DeviceMesh) -> Optimizer:
    if config.optim.name == "adamw":
        return AdamW(
            get_optimizer_params(model, config),
            lr=config.sched.lr,
            betas=config.optim.betas,
            eps=config.optim.eps,
            fused=True,
        )
    elif config.optim.name == "adamw_8bit":
        return AdamW8bit(
            get_optimizer_params(model, config),
            lr=config.sched.lr,
            betas=config.optim.betas,
            eps=config.optim.eps,
        )
    elif config.optim.name == "muon":
        return Muon(
            get_optimizer_params(model, config),
            lr=config.sched.lr,
            weight_decay=config.optim.weight_decay,
            distributed_mesh=mesh,
        )
    else:
        raise ValueError(f"Optimizer {config.name} not supported")
