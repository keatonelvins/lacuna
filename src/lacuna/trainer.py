"""Core training loop."""

import time
import torch
from loguru import logger
from torchtitan.tools import utils
from torch.distributed.elastic.multiprocessing.errors import record
from torchtitan.distributed.utils import dist_mean, clip_grad_norm_ as clip

from .checkpoint import save_checkpoint, load_checkpoint
from .config import LacunaConfig
from .data import LacunaDataset
from .scheduler import setup_scheduler
from .model import setup_model
from .optim import setup_optimizer
from .utils import display_config, log_training_metrics, setup_env, cleanup_env, setup_metrics_processor
from .wandb import init_wandb, log_wandb_metrics, finish
from .distributed import get_world_size, init_dist, setup_dist, destroy_dist


@record
@logger.catch(reraise=True)
def train(config: LacunaConfig) -> None:
    init_dist(config)
    run_dir = setup_env(config)
    display_config(config)

    wandb_run = init_wandb(config)
    world_size = get_world_size()

    logger.info(f"GPU setup: {world_size} GPUs")

    gc_handler = utils.GarbageCollection(gc_freq=10)

    try:
        logger.info("Setting up model")
        model = setup_model(config)
        model, amp_manager, mesh = setup_dist(model, config)

        logger.info("Setting up dataloader")
        dataset = LacunaDataset(config)
        logger.info(f"Packed dataset length: {dataset.length}")
        data_iter = iter(dataset.dataloader)

        if config.trainer.steps:
            total_steps = config.trainer.steps
        else:
            total_steps = dataset.length * config.trainer.epochs

        optimizer = setup_optimizer(model, config)
        scheduler = setup_scheduler(optimizer, config.scheduler, total_steps)
        metrics_processor = setup_metrics_processor(config, model)

        step, epoch = 0, 0

        if config.checkpoint.resume_from is not None:
            logger.info(f"Resuming from checkpoint: {config.checkpoint.resume_from}")
            load_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                dataloader=dataset.dataloader,
                path=config.checkpoint.resume_from,
            )

        logger.info("Starting training!")

        while step < total_steps:
            step += 1
            if step % dataset.length == 0:
                epoch += 1
                dataset.set_epoch(epoch)

            gc_handler.run(step)
            optimizer.zero_grad()

            data_load_start = time.perf_counter()
            batch = next(data_iter)
            ntokens_batch = batch["input_ids"].numel()
            metrics_processor.ntokens_since_last_log += ntokens_batch
            metrics_processor.data_loading_times.append(time.perf_counter() - data_load_start)

            labels = batch["input_ids"].clone()
            if "assistant_masks" in batch:
                labels[batch["assistant_masks"] == 0] = -100  # mask non-assistant tokens

            if config.model.compile_mode in ["reduce-overhead", "max-autotune"]:
                torch.compiler.cudagraph_mark_step_begin()

            model_inputs = {
                "input_ids": batch["input_ids"].cuda(),
                "position_ids": batch["position_ids"].cuda(),
                "labels": labels.cuda(),
                "accum_dtype": torch.float32,  # used by liger
            }

            with amp_manager:
                outputs = model(**model_inputs)
                loss = outputs.loss

            loss.backward()
            grad_norm = clip(model.parameters(), config.optimizer.max_norm)
            optimizer.step()
            scheduler.step()

            if step % config.metrics.log_every == 0:
                current_lr = scheduler.get_last_lr()[0]
                current_loss = dist_mean(loss.detach(), mesh) if mesh else loss.item()

                metrics_processor.update()
                log_training_metrics(step, current_loss, grad_norm, current_lr, run_dir)
                log_wandb_metrics(step, current_loss, grad_norm, current_lr, wandb_run)

            if config.checkpoint.save_every:
                if step > 0 and step % config.checkpoint.save_every == 0:
                    logger.info(f"Saving checkpoint at step {step}")
                    save_checkpoint(
                        step=step,
                        config=config,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        dataloader=dataset.dataloader,
                    )

        save_checkpoint(
            step=step,
            config=config,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            dataloader=dataset.dataloader,
            final=True,
        )
    except KeyboardInterrupt:
        logger.info("Training interrupted :(")
    finally:
        destroy_dist()
        finish(wandb_run)
        cleanup_env()
