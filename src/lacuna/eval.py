import torch
import torch.nn as nn
import math
from tqdm import tqdm
from torchtitan.distributed.utils import dist_sum
from torch.distributed.device_mesh import DeviceMesh

from lacuna.config import TrainConfig
from lacuna.data import PackedDataset


@torch.no_grad()
def run_eval(
    config: TrainConfig,
    model: nn.Module,
    amp_manager,
    mesh: DeviceMesh | None,
) -> dict[str, float]:
    """Calcuate eval metrics on held out data (perplexity, etc.)."""
    if not config.evals.datasets:
        return {}

    dataset = PackedDataset(config, mesh=mesh, train=False)
    data_iter = iter(dataset.dataloader)
    model.eval()

    device = torch.cuda.current_device()
    loss_sum = torch.zeros(1, dtype=torch.float64, device=device)
    token_sum = torch.zeros(1, dtype=torch.float64, device=device)

    for _ in tqdm(range(dataset.length), desc="Collecting eval metrics"):
        batch = next(data_iter)
        num_valid_tokens = batch["labels"].ne(-100).sum()

        model_inputs = {
            "input_ids": batch["input_ids"].cuda(),
            "position_ids": batch["position_ids"].cuda(),
            "labels": batch["labels"].cuda(),
            "accum_dtype": torch.float32,
            "skip_logits": True,
        }
        with amp_manager:
            loss = model(**model_inputs).loss

        loss_sum += loss.detach().to(torch.float64) * num_valid_tokens
        token_sum += num_valid_tokens

    if mesh:
        loss_mesh = mesh["dp"] if mesh.ndim > 1 else mesh

        loss_sum = dist_sum(loss_sum, loss_mesh)
        token_sum = dist_sum(token_sum, loss_mesh)

    mean_loss = (loss_sum / token_sum).item()
    perplexity = math.exp(mean_loss)

    return {
        "eval/loss": float(mean_loss),
        "eval/perplexity": float(perplexity),
        "eval/num_tokens": float(token_sum),
    }
