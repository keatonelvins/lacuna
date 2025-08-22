"""Core training loop for pretraining and SFT."""

import time
from typing import Any

import torch
from rich.pretty import pprint
from torch.amp import autocast
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from transformers import PreTrainedModel
from transformers.optimization import (
    get_cosine_with_min_lr_schedule_with_warmup,
    get_wsd_schedule,
)

from .checkpoint import cleanup_old_checkpoints, load_checkpoint, save_checkpoint
from .config import (
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

                peak_mfu = max(peak_mfu, mfu_metrics["mfu_pct"])
                peak_tflops = max(peak_tflops, mfu_metrics["tflops"])

                # Build log message with colors
                log_parts = [
                    f"\033[91mStep {step:>6}\033[0m",  # Red
                    f"\033[92mLoss: {accumulated_loss:.4f}\033[0m",  # Green
                    f"\033[93mGrad: {grad_norm:.4f}\033[0m",  # Yellow
                    f"\033[94mLR: {current_lr:.2e}\033[0m",  # Blue
                ]

                if "tokens_per_second" in mfu_metrics:
                    log_parts.append(
                        f"\033[96mTPS: {mfu_metrics['tokens_per_second']:,.0f}\033[0m"
                    )  # Cyan
                    log_parts.append(
                        f"\033[95mTFLOPS: {mfu_metrics['tflops']:.1f}\033[0m"
                    )  # Magenta
                    log_parts.append(
                        f"\033[92mMFU: {mfu_metrics['mfu_pct']:.1f}%\033[0m"
                    )  # Green

                log_parts.append(
                    f"\033[36mMem: {memory_stats['max_reserved_gb']:.1f}GB ({memory_stats['max_reserved_pct']:.0f}%)\033[0m"
                )  # Cyan

                # Add data loading time if significant
                if data_pct > 5:  # Only show if > 5% of time
                    log_parts.append(f"\033[33mData: {data_pct:.1f}%\033[0m")  # Yellow

                logger.info(" | ".join(log_parts))

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

        logger.info(
            f"All done! Total steps: {step}, Total tokens: {total_tokens:,}, "
            f"Peak MFU: {peak_mfu:.1f}%, Peak TFLOPS: {peak_tflops:.1f}"
        )
