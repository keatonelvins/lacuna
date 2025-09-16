"""Core training loop."""

import time
import torch
from loguru import logger
from torch.amp import autocast

from .checkpoint import save_checkpoint
from .config import LacunaConfig
from .data import setup_dataloader
from .scheduler import setup_scheduler
from .metrics import Redline
from .model import setup_model
from .optim import setup_optimizer
from .utils import display_config, log_training_metrics, setup_env, save_batch_json
from .wandb import init_wandb, log_wandb_metrics, finish
from .distributed import get_world_size, init_dist, setup_dist, destroy_dist


@logger.catch(reraise=True)
def train(config: LacunaConfig) -> None:
    init_dist(config)
    run_dir = setup_env(config)
    display_config(config)

    wandb_run = init_wandb(config)
    world_size = get_world_size()

    logger.info(f"GPU setup: {world_size} GPUs")

    try:
        logger.info("Setting up model")
        model = setup_model(config)
        model = setup_dist(model, config)

        logger.info("Setting up dataloader")
        dataloader, dataset = setup_dataloader(config)
        data_iter = iter(dataloader)

        if config.trainer.steps:
            total_steps = config.trainer.steps
        else:
            total_steps = dataset.length * config.trainer.epochs

        optimizer = setup_optimizer(model, config)
        scheduler = setup_scheduler(optimizer, config.scheduler, total_steps)
        redline = Redline(model=model, seq_len=config.trainer.seq_len, world_size=world_size)

        start_step = 0
        current_epoch = 0
        accum_dtype = torch.float32 if config.model.accum_fp32 else torch.bfloat16

        if config.checkpoint.resume_from is not None:
            logger.info(f"Resuming from checkpoint: {config.checkpoint.resume_from}")
            # TODO: Fix load_checkpoint to return step
            # start_step = load_checkpoint(
            #     model=model,
            #     optimizer=optimizer,
            #     scheduler=scheduler,
            #     dataloader=dataloader,
            #     path=config.checkpoint.resume_from,
            # )

        logger.info("Starting training!")

        for step in range(start_step, total_steps):
            if config.trainer.epochs > 1:
                epoch = step // dataset.length
                if epoch > current_epoch:
                    current_epoch = epoch
                    dataset.set_epoch(epoch)
            optimizer.zero_grad()

            data_load_start = time.perf_counter()
            batch = next(data_iter)
            data_load_time = time.perf_counter() - data_load_start

            if config.model.compile_mode in ["reduce-overhead", "max-autotune"]:
                torch.compiler.cudagraph_mark_step_begin()

            labels = batch["input_ids"].clone()
            labels[batch["position_ids"] == 0] = -100  # mask document boundaries

            if "assistant_masks" in batch:
                labels[batch["assistant_masks"] == 0] = -100  # mask non-assistant tokens

            batch["labels"] = labels
            model_inputs = {k: v.cuda() for k, v in batch.items()}
            model_inputs["accum_dtype"] = accum_dtype  # only used for Liger FLCE

            with autocast("cuda", dtype=torch.bfloat16):
                outputs = model(**model_inputs)
                loss = outputs.loss

            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.max_norm)
            if hasattr(grad_norm, "full_tensor"):  # needed for FSDP
                grad_norm = grad_norm.full_tensor()
            optimizer.step()
            scheduler.step()

            redline.update(batch["input_ids"].numel(), data_load_time)

            if step % config.metrics.steps_per_log == 0:
                current_lr = scheduler.get_last_lr()[0]
                metrics = redline.read()

                log_training_metrics(step, loss.item(), grad_norm, current_lr, metrics, run_dir)
                log_wandb_metrics(loss.item(), current_lr, grad_norm, step, metrics, wandb_run)
                save_batch_json(run_dir, step, batch)

            if config.checkpoint.save_every:
                interval = max(1, int(config.checkpoint.save_every * dataset.length))
                if step > 0 and step % interval == 0:
                    logger.info(f"Saving checkpoint at step {step}")
                    save_checkpoint(
                        step=step,
                        config=config,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        dataloader=dataloader,
                    )

        save_checkpoint(
            step=redline.step,
            config=config,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            dataloader=dataloader,
            final=True,
        )
    except KeyboardInterrupt:
        logger.info("Training interrupted :(")
    finally:
        finish(wandb_run)
        destroy_dist()
