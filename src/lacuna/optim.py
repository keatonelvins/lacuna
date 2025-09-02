from torch.optim import Optimizer, AdamW
from transformers import PreTrainedModel

from .config import PretrainConfig, SFTConfig


def setup_optimizer(
    model: PreTrainedModel, config: PretrainConfig | SFTConfig
) -> Optimizer:
    """Setup AdamW optimizer."""
    return AdamW(
        model.parameters(),
        lr=config.optimizer.lr,
        weight_decay=config.optimizer.weight_decay,
        betas=config.optimizer.betas,
        fused=True,
    )
