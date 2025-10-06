"""Core training loop."""

import time
import torch
from loguru import logger
from torchtitan.tools import utils
from torch.distributed.elastic.multiprocessing.errors import record
from torchtitan.distributed.utils import dist_mean, clip_grad_norm_ as clip

from lacuna.checkpoint import save_checkpoint, load_checkpoint
from lacuna.config import LacunaConfig
from lacuna.data import LacunaDataset
from lacuna.scheduler import setup_scheduler
from lacuna.model import setup_model
from lacuna.optim import setup_optimizer
from lacuna.utils import setup_env, cleanup_env, log_training_metrics, setup_metrics_processor, log_eval_metrics, log_loss_spikes
from lacuna.wandb import init_wandb, log_wandb_metrics, finish
from lacuna.distributed import init_dist, setup_dist, destroy_dist
from lacuna.eval import run_eval, run_vf_envs


@record
@logger.catch(reraise=True)
def train(config: LacunaConfig) -> None:
    init_dist(config)
    run_dir = setup_env(config)
    wandb_run = init_wandb(config)

    gc_handler = utils.GarbageCollection()

    try:
        model = setup_model(config)
        model, amp_manager, mesh = setup_dist(model, config)

        logger.info("Setting up dataloader")
        dataset = LacunaDataset(config)
        logger.info(f"Packed dataset length: {dataset.length}")

        if config.trainer.steps:
            total_steps = config.trainer.steps
        else:
            total_steps = dataset.length * config.trainer.epochs

        optimizer = setup_optimizer(model, config)
        scheduler = setup_scheduler(optimizer, config.scheduler, total_steps)
        metrics_processor = setup_metrics_processor(config, model)

        step = 0
        prev_loss = None

        if config.checkpoint.resume_from:
            step = load_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler if config.checkpoint.full_state else None,
                dataloader=dataset.dataloader if config.checkpoint.full_state else None,
                path=config.checkpoint.resume_from,
            )
            if config.checkpoint.full_state:
                logger.info(f"Resumed from step {step} (full state)")
            else:
                logger.info("Loaded model+optimizer from checkpoint, setting step=0")
                step = 0

        logger.info(f"Starting training at step {step + 1}")

        data_iter = iter(dataset.dataloader)

        while step < total_steps:
            step += 1
            epoch = step // dataset.length

            if step % dataset.length == 1:
                dataset.set_epoch(epoch)
                data_iter = iter(dataset.dataloader)

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
                loss = model(**model_inputs).loss

            loss.backward()
            grad_norm = clip(model.parameters(), config.optimizer.max_norm)
            optimizer.step()
            scheduler.step()

            local_loss = loss.detach().item()
            if prev_loss is not None and local_loss > prev_loss * 3.0:
                log_loss_spikes(step, local_loss, model_inputs, run_dir)
            prev_loss = local_loss

            if step % config.metrics.log_every == 0:
                current_lr = scheduler.get_last_lr()[0]
                current_grad_norm = grad_norm.item()  # already reduced
                current_loss = dist_mean(loss.detach(), mesh) if mesh else local_loss

                metrics = {
                    "train/loss": current_loss,
                    "train/lr": current_lr,
                    "train/grad_norm": current_grad_norm,
                    "train/ntokens_micro_batch": ntokens_batch,
                    **metrics_processor.get_metrics(),
                }
                log_training_metrics(step, metrics, run_dir)
                log_wandb_metrics(step, metrics, wandb_run)

            if config.checkpoint.save_every and step % config.checkpoint.save_every == 0:
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
        )

        if config.evals.datasets:
            logger.info("Running eval")
            eval_metrics = run_eval(config, model, amp_manager, mesh)
            eval_metrics.update(run_vf_envs(config))  # TODO: clear model from GPU first
            log_eval_metrics(step, eval_metrics, run_dir)
            log_wandb_metrics(step, eval_metrics, wandb_run)

    except KeyboardInterrupt:
        logger.info("Training interrupted :(")
    finally:
        destroy_dist()
        finish(wandb_run)
        cleanup_env()
