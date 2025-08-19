"""Core training loop for pretraining and SFT."""

import time
from typing import Any

import torch
from rich.pretty import pprint
from torch.amp import autocast
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from transformers import PreTrainedModel, get_cosine_schedule_with_warmup

from .checkpoint import cleanup_old_checkpoints, load_checkpoint, save_checkpoint
from .config import ModelConfig, PretrainConfig, SFTConfig
from .data import setup_dataloader
from .distributed import get_world_size, init_distributed, setup_fsdp
from .utils import setup_logger
from loguru import logger


def setup_model(config: ModelConfig) -> PreTrainedModel:
    """Load model with flash attention and liger kernels."""
    return AutoLigerKernelForCausalLM.from_pretrained(
        config.name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        use_cache=False,
    )


def setup_optimizer(model: PreTrainedModel, config: Any) -> torch.optim.AdamW:
    """Setup AdamW optimizer."""
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.optimizer.lr,
        weight_decay=config.optimizer.weight_decay,
        betas=config.optimizer.betas,
    )


def train(config: PretrainConfig | SFTConfig) -> None:
    """Core training function."""
    setup_logger()

    logger.info("Starting training with config:")
    # TODO: merge into logger
    pprint(config, expand_all=True)  # omg Will you've outdone yourself

    init_distributed()

    # high -> TF32, highest -> FP32
    torch.set_float32_matmul_precision("high")

    world_size = get_world_size()
    batch_size = config.trainer.batch_size

    if not batch_size % world_size == 0:
        raise ValueError(
            f"Batch size {batch_size} must be divisible by world_size {world_size}"
        )

    micro_batch_size = batch_size // world_size

    logger.info(
        f"GPU setup: {world_size} GPUs, batch_size={batch_size} ({micro_batch_size} per GPU)"
    )

    logger.info(f"Loading model: {config.model.name}")
    model = setup_model(config.model)

    model = model.cuda()
    model.train()

    # Apply FSDP if enabled and multi-GPU
    if config.fsdp.enabled and world_size > 1:
        model = setup_fsdp(
            model,
            reshard_after_forward=config.fsdp.reshard_after_forward,
            cpu_offload=config.fsdp.cpu_offload,
            sharding_strategy=config.fsdp.sharding_strategy,
        )

    logger.info("Setting up optimizer and scheduler")
    optimizer = setup_optimizer(model, config)

    logger.info("Setting up dataloader")
    dataloader = setup_dataloader(config, micro_batch_size)

    if isinstance(config, PretrainConfig):
        max_steps = config.trainer.steps
    else:  # SFTConfig
        max_steps = len(dataloader) * config.trainer.epochs

    # TODO: support linear, wsd, etc.
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.scheduler.warmup_steps,
        num_training_steps=max_steps,
    )

    start_step = 0
    total_tokens = 0

    # Resume from checkpoint if specified
    if config.checkpoint.resume_path is not None:
        logger.info(f"Resuming from checkpoint: {config.checkpoint.resume_path}")
        training_state = load_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            path=config.checkpoint.resume_path,
        )
        start_step = training_state["step"]
        total_tokens = training_state["total_tokens"]
        logger.info(f"Resumed from step {start_step}, total tokens: {total_tokens:,}")

    logger.info(f"Starting training: {max_steps} steps (current step: {start_step})")

    try:
        dataloader_iter = iter(dataloader)

        for step in range(start_step, max_steps):
            step_start_time = time.time()
            accumulated_loss = 0.0
            optimizer.zero_grad()

            try:
                batch = next(dataloader_iter)
            except StopIteration:
                # TODO: should we flag for pt?
                dataloader_iter = iter(dataloader)
                batch = next(dataloader_iter)

            model_inputs = {k: v.cuda() for k, v in batch.items()}

            with autocast("cuda", dtype=torch.bfloat16):
                outputs = model(**model_inputs)
                loss = outputs.loss

            loss.backward()
            accumulated_loss += loss.item()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.optimizer.grad_clip
            )
            optimizer.step()
            scheduler.step()

            step_tokens = batch_size * config.data.seq_len
            total_tokens += step_tokens

            if step % config.metrics.log_every == 0:
                step_time = time.time() - step_start_time
                tokens_per_sec = step_tokens / step_time if step_time > 0 else 0
                current_lr = scheduler.get_last_lr()[0]
                memory_gb = torch.cuda.max_memory_allocated() / 1024**3

                logger.info(
                    f"Step {step:>6} | Loss: {accumulated_loss:.4f} | "
                    f"Grad Norm: {grad_norm:.4f} | LR: {current_lr:.2e} | "
                    f"Tokens/s: {tokens_per_sec:>6.0f} | Memory: {memory_gb:.1f}GB"
                )

                torch.cuda.reset_peak_memory_stats()

            if step > 0 and step % config.checkpoint.save_every == 0:
                logger.info(f"Saving checkpoint at step {step}")
                checkpoint_path = config.checkpoint.save_dir / f"step_{step}"
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                    total_tokens=total_tokens,
                    path=checkpoint_path,
                )
                cleanup_old_checkpoints(
                    config.checkpoint.save_dir, config.checkpoint.keep_latest
                )

    except KeyboardInterrupt:
        logger.info("Training interrupted!!!")

    finally:
        logger.info("Saving final checkpoint")
        final_path = config.checkpoint.save_dir / "final"
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=step,
            total_tokens=total_tokens,
            path=final_path,
        )

        logger.info(f"All done! Total steps: {step}, Total tokens: {total_tokens:,}")
