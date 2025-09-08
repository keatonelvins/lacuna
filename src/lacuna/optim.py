import re
from torch import nn
from torch.optim import Optimizer, AdamW
from transformers import PreTrainedModel
from transformers.trainer_pt_utils import get_parameter_names

from .config import LacunaConfig

# ref: https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py
def get_decay_parameter_names(model) -> list[str]:
    exclude_params = [r"bias", r"layernorm", r"rmsnorm", r"(?:^|\.)norm(?:$|\.)", r"_norm(?:$|\.)"]
    exclude_params.extend([r"embed_tokens", r"wte", r"wpe", r"embeddings"])  # also exclude embeddings!
    decay_parameters = get_parameter_names(model, [nn.LayerNorm], exclude_params)
    return decay_parameters


def get_optimizer_params(model, config: LacunaConfig) -> list[dict]:
    """Split params based on if weight decay should be applied."""
    decay_names = get_decay_parameter_names(model)
    decay_params = [p for n, p in model.named_parameters() if p.requires_grad and n in decay_names]
    no_decay_params = [p for n, p in model.named_parameters() if p.requires_grad and n not in decay_names]
    optimizer_params = [
        {"params": decay_params, "weight_decay": config.optimizer.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return optimizer_params


def setup_optimizer(model: PreTrainedModel, config: LacunaConfig) -> Optimizer:
    """Setup AdamW optimizer."""    
    return AdamW(
        get_optimizer_params(model, config),
        lr=config.optimizer.lr,
        betas=config.optimizer.betas,
        fused=True,
    )
