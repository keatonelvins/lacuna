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
from .distributed import get_world_size, init_distributed, setup_distributed, is_master, destroy_distributed
from .metrics import Redline
from .model import setup_model
from .optim import setup_optimizer
from .utils import setup_logger, display_config, log_training_metrics, setup_env
from .wandb import init_wandb, log_metrics, prepare_wandb_metrics, finish


@logger.catch(reraise=True)
def train(config: PretrainConfig | SFTConfig) -> None:
    """Core training function."""
    setup_logger()
    setup_env()

    init_distributed()

    if is_master():
        display_config(config)

    # high -> TF32, highest -> FP32
    torch.set_float32_matmul_precision("high")

    wandb_run = init_wandb(config)
    config.checkpoint.prepare_save_dir()  # clear save_dir if not resuming

    world_size = get_world_size()

    if not config.trainer.batch_size % world_size == 0:
        raise ValueError(f"Batch size {config.trainer.batch_size} must be divisible by world_size {world_size}")

    micro_batch_size = config.trainer.batch_size // world_size

    logger.info(f"GPU setup: {world_size} GPUs, batch_size={config.trainer.batch_size} ({micro_batch_size} per GPU)")

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
        raise ValueError("SFTConfig is not supported")

    redline = Redline(
        model=model,
        seq_len=config.data.seq_len,
        world_size=world_size,
    )

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

    if config.checkpoint.resume_from is not None:
        logger.info(f"Resuming from checkpoint: {config.checkpoint.resume_from}")
        redline.state = load_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            path=config.checkpoint.resume_from,
        )
        logger.info(f"Resumed from checkpoint: {config.checkpoint.resume_from}")

    logger.info(f"Starting training: {max_steps} steps")

    try:
        dataloader_iter = iter(dataloader)
        start_step = redline.state.step

        for step in range(start_step, max_steps):
            optimizer.zero_grad()

            data_load_start = time.perf_counter()
            batch = next(dataloader_iter)
            data_load_time = time.perf_counter() - data_load_start

            if config.model.compile_mode in ["reduce-overhead", "max-autotune"]:
                torch.compiler.cudagraph_mark_step_begin()

            model_inputs = {k: v.cuda() for k, v in batch.items()}
            model_inputs["accum_dtype"] = accum_dtype  # only used for Liger FLCE

            with autocast("cuda", dtype=torch.bfloat16):
                outputs = model(**model_inputs)
                loss = outputs.loss

            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.grad_clip)
            if hasattr(grad_norm, "full_tensor"):  # TODO: read FSDP docs to see if this is correct
                grad_norm = grad_norm.full_tensor()
            optimizer.step()
            scheduler.step()

            step_tokens = config.trainer.batch_size * config.data.seq_len
            redline.update(step_tokens, data_load_time)

            if step % config.metrics.log_every == 0:
                current_lr = scheduler.get_last_lr()[0]
                metrics = redline.read()

                log_training_metrics(step, loss.item(), grad_norm, current_lr, metrics)

                if wandb_run:
                    wandb_metrics = prepare_wandb_metrics(loss.item(), grad_norm, current_lr, metrics, redline.state)
                    log_metrics(wandb_metrics, step, wandb_run)

            if step > 0 and config.checkpoint.save_every and step % config.checkpoint.save_every == 0:
                logger.info(f"Saving checkpoint at step {step}")
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    state=redline.state,
                    config=config,
                    dataloader=dataloader,
                    final=False,
                    tokenizer=tokenizer,
                )

    except KeyboardInterrupt:
        logger.info("Training interrupted!!!")

    finally:
        if redline.state.step > start_step:  # don't save if training insta-crashed
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                state=redline.state,
                dataloader=dataloader,
                tokenizer=tokenizer,
                final=True,
            )

        finish(wandb_run)
        destroy_distributed()
