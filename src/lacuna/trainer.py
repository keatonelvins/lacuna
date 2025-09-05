"""Core training loop for pretraining and SFT."""

import time
import torch
from loguru import logger
from torch.amp import autocast
from transformers.optimization import (
    get_cosine_with_min_lr_schedule_with_warmup,
    get_wsd_schedule,
)

from .checkpoint import (
    cleanup_old_checkpoints,
    load_checkpoint,
    save_checkpoint,
)
from .config import (
    CosineSchedulerConfig,
    WSDSchedulerConfig,
    PretrainConfig,
    SFTConfig,
)
from .data import setup_dataloader
from .distributed import get_world_size, init_distributed, setup_distributed
from .metrics import MFUTracker, MemoryTracker, StateTracker
from .model import setup_model
from .optim import setup_optimizer
from .utils import setup_logger, display_config
from .wandb import init_wandb, log_metrics, finish


@logger.catch(reraise=True)
def train(config: PretrainConfig | SFTConfig) -> None:
    """Core training function."""
    setup_logger()
    display_config(config)

    init_distributed()

    # high -> TF32, highest -> FP32
    torch.set_float32_matmul_precision("high")

    wandb_run = init_wandb(config)
    config.checkpoint.prepare_save_dir()  # clear save_dir if not resuming

    world_size = get_world_size()
    batch_size = config.trainer.batch_size

    if not batch_size % world_size == 0:
        raise ValueError(f"Batch size {batch_size} must be divisible by world_size {world_size}")

    micro_batch_size = batch_size // world_size

    logger.info(f"GPU setup: {world_size} GPUs, batch_size={batch_size} ({micro_batch_size} per GPU)")

    logger.info("Setting up model")
    model = setup_model(config)
    model = setup_distributed(model, config)

    logger.info("Setting up optimizer and scheduler")
    optimizer = setup_optimizer(model, config)

    logger.info("Setting up dataloader")
    dataloader, tokenizer = setup_dataloader(config, micro_batch_size)

    if isinstance(config, PretrainConfig):
        max_steps = config.trainer.steps
    else:  # SFTConfig
        max_steps = len(dataloader) * config.trainer.epochs

    state = StateTracker()
    mfu_tracker = MFUTracker(
        model=model,
        seq_len=config.data.seq_len,
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

    accum_dtype = torch.float32 if config.model.accum_fp32 else torch.bfloat16

    if config.checkpoint.resume_path is not None:
        logger.info(f"Resuming from checkpoint: {config.checkpoint.resume_path}")
        state = load_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            path=config.checkpoint.resume_path,
        )
        logger.info(f"Resumed from checkpoint: {config.checkpoint.resume_path}")

    logger.info(f"Starting training: {max_steps} steps")

    # Track data loading times and start time
    data_loading_times = []
    start_time = time.perf_counter()

    try:
        dataloader_iter = iter(dataloader)
        start_step = state.step

        for step in range(start_step, max_steps):
            accumulated_loss = 0.0
            optimizer.zero_grad()

            # Track data loading time
            data_load_start = time.perf_counter()
            batch = next(dataloader_iter)
            data_loading_times.append(time.perf_counter() - data_load_start)

            if config.model.compile_mode in ["reduce-overhead", "max-autotune"]:
                torch.compiler.cudagraph_mark_step_begin()

            model_inputs = {k: v.cuda() for k, v in batch.items()}
            model_inputs["accum_dtype"] = accum_dtype  # only used for Liger FLCE

            with autocast("cuda", dtype=torch.bfloat16):
                outputs = model(**model_inputs)
                loss = outputs.loss

            loss.backward()
            accumulated_loss += loss.item()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.grad_clip)
            optimizer.step()
            scheduler.step()

            step_tokens = batch_size * config.data.seq_len
            state.total_tokens += step_tokens

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
                    state.peak_mfu = max(state.peak_mfu, mfu_metrics["mfu_pct"])
                    state.peak_tflops = max(state.peak_tflops, mfu_metrics["tflops"])

                state.peak_mem_gb = max(state.peak_mem_gb, memory_stats["max_reserved_gb"])

                log_parts = [
                    f"\033[91mStep {step:>6}\033[0m",
                    f"\033[92mLoss: {accumulated_loss:7.4f}\033[0m",
                    f"\033[93mGrad: {grad_norm:8.4f}\033[0m",
                    f"\033[94mLR: {current_lr:9.2e}\033[0m",
                    f"\033[36mMem: {memory_stats['max_reserved_gb']:5.1f}GB ({memory_stats['max_reserved_pct']:3.0f}%)\033[0m",
                ]

                if "mfu_pct" in mfu_metrics:
                    log_parts.append(f"\033[92mMFU: {mfu_metrics['mfu_pct']:5.1f}%\033[0m")

                if data_pct > 5:  # Only show if > 5% of wall-clock time
                    log_parts.append(f"\033[33mData: {data_pct:5.1f}%\033[0m")

                logger.info(" | ".join(log_parts))

                if wandb_run:
                    wandb_metrics = {
                        "train/loss": accumulated_loss,
                        "train/grad_norm": grad_norm,
                        "train/lr": current_lr,
                        "train/total_tokens": state.total_tokens,
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
                                "perf/peak_mfu_pct": state.peak_mfu,
                                "perf/peak_tflops": state.peak_tflops,
                            }
                        )

                    log_metrics(wandb_metrics, step, wandb_run)

                memory_tracker.reset_peak_stats()
                data_loading_times.clear()

            if step > 0 and step % config.checkpoint.save_every == 0:
                logger.info(f"Saving checkpoint at step {step} (peak MFU: {state.peak_mfu:.1f}%, peak memory: {state.peak_mem_gb:.1f}GB)")
                checkpoint_path = config.checkpoint.save_dir / f"step_{step}"
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    state=state,
                    path=checkpoint_path,
                    config=config,
                    dataloader=None,
                    final=False,
                    tokenizer=tokenizer,
                )
                cleanup_old_checkpoints(config.checkpoint.save_dir, config.checkpoint.keep_latest)

            state.step += 1

    except KeyboardInterrupt:
        logger.info("Training interrupted!!!")

    finally:
        logger.info("Saving final checkpoint")
        final_path = config.checkpoint.save_dir / "final"
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            path=final_path,
            config=config,
            state=state,
            dataloader=None,
            tokenizer=tokenizer,
            final=True,  # Final checkpoint in HF format
        )

        finish(wandb_run)
