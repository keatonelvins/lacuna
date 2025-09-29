"""Evaluation loop for trainer."""

import math
import torch
from torchtitan.distributed.utils import dist_sum
from torch.distributed.device_mesh import DeviceMesh
from transformers import PreTrainedModel

from lacuna.config import LacunaConfig
from lacuna.data import LacunaDataset


def run_eval(
    config: LacunaConfig,
    model: PreTrainedModel,
    amp_manager,
    mesh: DeviceMesh | None,
) -> dict[str, float]:
    if not config.evals.datasets:
        return {}

    dataset = LacunaDataset(config, datasets=config.evals.datasets)
    data_iter = iter(dataset.dataloader)

    was_training = model.training
    model.eval()

    batch_count = 0
    loss_sum = torch.zeros(1, dtype=torch.float64)
    token_sum = torch.zeros(1, dtype=torch.float64)
    correct_sum = torch.zeros(1, dtype=torch.float64)

    for _ in range(dataset.length):
        batch = next(data_iter)
        labels = batch["input_ids"].clone()
        if "assistant_masks" in batch:
            labels[batch["assistant_masks"] == 0] = -100

        model_inputs = {
            "input_ids": batch["input_ids"].cuda(),
            "position_ids": batch["position_ids"].cuda(),
            "labels": labels.cuda(),
        }

        labels_gpu = model_inputs["labels"]

        with torch.no_grad():
            with amp_manager:
                outputs = model(**model_inputs)

        loss = outputs.loss.detach().to(torch.float64)
        logits = outputs.logits

        mask = labels_gpu.ne(-100)
        valid_tokens = mask.sum()
        if valid_tokens.item() == 0:
            continue

        token_count = valid_tokens.to(torch.float64)
        loss_value = loss * token_count
        predictions = logits.argmax(dim=-1)
        correct_tokens = (predictions.eq(labels_gpu) & mask).sum().to(torch.float64)

        loss_sum += loss_value
        token_sum += token_count
        correct_sum += correct_tokens
        batch_count += 1

    if was_training:
        model.train()

    if mesh:
        total_loss = dist_sum(loss_sum, mesh)
        total_tokens = dist_sum(token_sum, mesh)
        total_correct = dist_sum(correct_sum, mesh)
        total_batches = dist_sum(
            torch.tensor([batch_count], device=loss_sum.device, dtype=torch.float64),
            mesh,
        )
    else:
        total_loss = loss_sum.item()
        total_tokens = token_sum.item()
        total_correct = correct_sum.item()
        total_batches = float(batch_count)

    if total_tokens == 0:
        return {}

    mean_loss = total_loss / total_tokens
    perplexity = math.exp(mean_loss)
    token_accuracy = total_correct / total_tokens

    return {
        "eval/loss": float(mean_loss),
        "eval/perplexity": float(perplexity),
        "eval/token_accuracy": float(token_accuracy),
        "eval/num_tokens": float(total_tokens),
        "eval/num_batches": float(total_batches),
    }
