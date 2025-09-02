from torch.optim import Optimizer, AdamW
from transformers import PreTrainedModel

from .config import LacunaConfig


def setup_optimizer(model: PreTrainedModel, config: LacunaConfig) -> Optimizer:
    """Setup AdamW optimizer."""
    return AdamW(
        model.parameters(),
        lr=config.optimizer.lr,
        weight_decay=config.optimizer.weight_decay,
        betas=config.optimizer.betas,
        fused=True,
    )
