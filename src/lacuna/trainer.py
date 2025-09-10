"""Core training loop."""

import time
import torch
from loguru import logger
from torch.amp import autocast

from .checkpoint import (
    load_checkpoint,
    save_checkpoint,
)
from .config import LacunaConfig
from .data import setup_dataloader
from .scheduler import setup_scheduler
from .distributed import get_world_size, init_distributed, setup_distributed, destroy_distributed
from .metrics import Redline
from .model import setup_model
from .optim import setup_optimizer
from .utils import setup_logger, display_config, log_training_metrics, setup_env
from .wandb import init_wandb, log_metrics, prepare_wandb_metrics, finish


@logger.catch(reraise=True)
def train(config: LacunaConfig) -> None:
    """Core training function."""
    setup_logger()
    setup_env()

    init_distributed()
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

    logger.info("Setting up dataloader")
    dataloader, tokenizer, dataset = setup_dataloader(config, micro_batch_size)

    if config.trainer.steps:
        total_steps = config.trainer.steps
    else:  # must be map-style
        total_steps = dataset.length * config.trainer.epochs

    logger.info("Setting up optimizer and scheduler")
    optimizer = setup_optimizer(model, config)
    scheduler = setup_scheduler(
        optimizer,
        config.scheduler,
        total_steps=total_steps,
    )

    redline = Redline(
        model=model,
        seq_len=config.trainer.seq_len,
        world_size=world_size,
    )

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

    logger.info(f"Starting training: {total_steps} steps")

    try:
        dataloader_iter = iter(dataloader)
        start_step = redline.state.step
        current_epoch = 0

        for step in range(start_step, total_steps):
            if dataset.length and config.trainer.epochs > 1:
                epoch = step // dataset.length
                if epoch > current_epoch:
                    current_epoch = epoch
                    dataset.set_epoch(epoch)
                    dataloader_iter = iter(dataloader)
            optimizer.zero_grad()

            data_load_start = time.perf_counter()
            batch = next(dataloader_iter)
            data_load_time = time.perf_counter() - data_load_start

            if config.model.compile_mode in ["reduce-overhead", "max-autotune"]:
                torch.compiler.cudagraph_mark_step_begin()

            batch["labels"] = batch["input_ids"].clone()
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

            step_tokens = batch["input_ids"].shape[1]
            redline.update(step_tokens, data_load_time)

            if step % config.metrics.steps_per_log == 0:
                current_lr = scheduler.get_last_lr()[0]
                metrics = redline.read()

                log_training_metrics(step, loss.item(), grad_norm, current_lr, metrics)

                if wandb_run:
                    wandb_metrics = prepare_wandb_metrics(loss.item(), grad_norm, current_lr, metrics, redline.state)
                    log_metrics(wandb_metrics, step, wandb_run)

            if current_epoch > 0 and step % int(config.checkpoint.save_every * dataset.length) == 0:
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
