"""Core training loop for pretraining and SFT."""

import time
from typing import Any

import os
from contextlib import redirect_stdout, redirect_stderr

import torch
from rich.pretty import Pretty
from rich.console import Console
from torch.amp import autocast
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from transformers import PreTrainedModel
from transformers.optimization import (
    get_cosine_with_min_lr_schedule_with_warmup,
    get_wsd_schedule,
)

from .checkpoint import cleanup_old_checkpoints, load_checkpoint, save_checkpoint
from .config import (
    CompileConfig,
    CosineSchedulerConfig,
    WSDSchedulerConfig,
    ModelConfig,
    PretrainConfig,
    SFTConfig,
)
from .data import setup_dataloader
from .distributed import get_world_size, init_distributed, setup_fsdp
from .metrics import MFUTracker, MemoryTracker
from .utils import setup_logger
from .wandb import init_wandb, log_metrics, finish
from loguru import logger


def setup_model(config: ModelConfig) -> PreTrainedModel:
    """Load model with flash attention and liger kernels."""
    logger.info(f"Loading model: {config.name}")
    with (
        open(os.devnull, "w") as devnull,
        redirect_stdout(devnull),
        redirect_stderr(devnull),
    ):  # silence unaesthetic liger print (lol liger print)
        model = AutoLigerKernelForCausalLM.from_pretrained(
            config.name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            use_cache=False,
        )
    return model


def setup_optimizer(model: PreTrainedModel, config: Any) -> torch.optim.AdamW:
    """Setup AdamW optimizer."""
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.optimizer.lr,
        weight_decay=config.optimizer.weight_decay,
        betas=config.optimizer.betas,
    )


def apply_torch_compile(
    model: PreTrainedModel, compile_config: CompileConfig
) -> PreTrainedModel:
    """Apply torch.compile to each individual transformer block."""
    if not compile_config.enabled:
        return model

    # Set dynamo configs for stability
    if hasattr(torch, "_dynamo"):
        torch._dynamo.config.cache_size_limit = 256
        torch._dynamo.config.suppress_errors = True

    layers = model.model.layers
    for idx, layer in enumerate(layers):
        compiled_layer = torch.compile(
            layer,
            fullgraph=compile_config.fullgraph,
            mode=compile_config.mode,
        )
        layers[idx] = compiled_layer
    logger.info(f"Applied torch.compile (mode={compile_config.mode})")

    return model


def train(config: PretrainConfig | SFTConfig) -> None:
    """Core training function."""
    setup_logger()

    console = Console()
    with console.capture() as capture:
        console.print(
            Pretty(config, expand_all=True)
        )  # omg Will you've outdone yourself
    logger.info("Starting training with config:\n" + capture.get().strip())

    init_distributed()

    # high -> TF32, highest -> FP32
    torch.set_float32_matmul_precision("high")

    wandb_run = init_wandb(config)

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

    try:
        model = setup_model(config.model)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise e

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

    model = apply_torch_compile(model, config.compile)

    logger.info("Setting up optimizer and scheduler")
    optimizer = setup_optimizer(model, config)

    logger.info("Setting up dataloader")
    dataloader = setup_dataloader(config, micro_batch_size)

    if isinstance(config, PretrainConfig):
        max_steps = config.trainer.steps
    else:  # SFTConfig
        max_steps = len(dataloader) * config.trainer.epochs

    # Setup MFU and memory tracking
    mfu_tracker = MFUTracker(
        model=model,
        seq_len=config.data.seq_len,
        window_size=10,
        world_size=world_size,
    )
    memory_tracker = MemoryTracker()

    if isinstance(config.scheduler, CosineSchedulerConfig):
        scheduler = get_cosine_with_min_lr_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.scheduler.warmup_ratio * max_steps,
            num_training_steps=max_steps,
            min_lr_ratio=config.scheduler.min_lr_ratio,
        )
    elif isinstance(config.scheduler, WSDSchedulerConfig):
        scheduler = get_wsd_schedule(
            optimizer,
            num_warmup_steps=config.scheduler.warmup_steps,
            num_decay_steps=config.scheduler.decay_steps,
            num_training_steps=max_steps,
            min_lr_ratio=config.scheduler.min_lr_ratio,
            decay_type=config.scheduler.decay_type,
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {config.scheduler.type}")

    start_step = 0
    total_tokens = 0
    peak_mfu = 0.0
    peak_tflops = 0.0

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
        peak_mfu = training_state.get("peak_mfu", 0.0)
        peak_tflops = training_state.get("peak_tflops", 0.0)
        logger.info(
            f"Resumed from step {start_step}, total tokens: {total_tokens:,}, "
            f"peak MFU: {peak_mfu:.1f}%, peak TFLOPS: {peak_tflops:.1f}"
        )

    logger.info(f"Starting training: {max_steps} steps (current step: {start_step})")

    # Track data loading times and start time
    data_loading_times = []
    start_time = time.perf_counter()

    try:
        dataloader_iter = iter(dataloader)

        for step in range(start_step, max_steps):
            accumulated_loss = 0.0
            optimizer.zero_grad()

            # Track data loading time
            data_load_start = time.perf_counter()
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                # TODO: should we flag for pt?
                dataloader_iter = iter(dataloader)
                batch = next(dataloader_iter)
            data_loading_times.append(time.perf_counter() - data_load_start)

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

            mfu_tracker.update(step_tokens)

            if step % config.metrics.log_every == 0:
                current_lr = scheduler.get_last_lr()[0]

                mfu_metrics = mfu_tracker.get_metrics()
                memory_stats = memory_tracker.get_memory_stats()

                if data_loading_times:
                    total_time = time.perf_counter() - start_time
                    data_pct = 100 * sum(data_loading_times) / max(total_time, 0.001)
                else:
                    data_pct = 0

                if "mfu_pct" in mfu_metrics:
                    peak_mfu = max(peak_mfu, mfu_metrics["mfu_pct"])
                    peak_tflops = max(peak_tflops, mfu_metrics["tflops"])

                log_parts = [
                    f"\033[91mStep {step:>6}\033[0m",
                    f"\033[92mLoss: {accumulated_loss:7.4f}\033[0m",
                    f"\033[93mGrad: {grad_norm:8.4f}\033[0m",
                    f"\033[94mLR: {current_lr:9.2e}\033[0m",
                    f"\033[36mMem: {memory_stats['max_reserved_gb']:5.1f}GB ({memory_stats['max_reserved_pct']:3.0f}%)\033[0m",
                ]

                if "mfu_pct" in mfu_metrics:
                    log_parts.append(
                        f"\033[92mMFU: {mfu_metrics['mfu_pct']:5.1f}%\033[0m"
                    )

                if data_pct > 5:  # Only show if > 5% of wall-clock time
                    log_parts.append(f"\033[33mData: {data_pct:5.1f}%\033[0m")

                logger.info(" | ".join(log_parts))

                if wandb_run:
                    wandb_metrics = {
                        "train/loss": accumulated_loss,
                        "train/grad_norm": grad_norm,
                        "train/lr": current_lr,
                        "train/total_tokens": total_tokens,
                        "perf/data_loading_pct": data_pct,
                        "memory/max_reserved_gb": memory_stats["max_reserved_gb"],
                        "memory/max_reserved_pct": memory_stats["max_reserved_pct"],
                    }

                    if "tps" in mfu_metrics:
                        wandb_metrics.update(
                            {
                                "perf/tps": mfu_metrics["tps"],
                                "perf/tflops": mfu_metrics["tflops"],
                                "perf/mfu_pct": mfu_metrics["mfu_pct"],
                                "perf/peak_mfu_pct": peak_mfu,
                                "perf/peak_tflops": peak_tflops,
                            }
                        )

                    log_metrics(wandb_metrics, step, wandb_run)

                memory_tracker.reset_peak_stats()
                data_loading_times.clear()

            if step > 0 and step % config.checkpoint.save_every == 0:
                logger.info(
                    f"Saving checkpoint at step {step} (peak MFU: {peak_mfu:.1f}%)"
                )
                checkpoint_path = config.checkpoint.save_dir / f"step_{step}"
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                    total_tokens=total_tokens,
                    path=checkpoint_path,
                    peak_mfu=peak_mfu,
                    peak_tflops=peak_tflops,
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
            peak_mfu=peak_mfu,
            peak_tflops=peak_tflops,
        )

        finish(wandb_run)

        logger.info(
            f"All done! Total steps: {step}, Total tokens: {total_tokens:,}, "
            f"Peak MFU: {peak_mfu:.1f}%, Peak TFLOPS: {peak_tflops:.1f}"
        )
