"""Core training loop."""

import time
import torch
from loguru import logger
from torch.amp import autocast
from torch.distributed.elastic.multiprocessing.errors import record
from torchtitan.tools import utils
from torchtitan.components.metrics import build_device_memory_monitor

from .checkpoint import save_checkpoint
from .config import LacunaConfig
from .data import LacunaDataset
from .scheduler import setup_scheduler
from .metrics import Redline, calculate_model_flops
from .model import setup_model
from .optim import setup_optimizer
from .utils import display_config, log_training_metrics, setup_env, cleanup_env
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
        model = setup_dist(model, config)

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
        redline = Redline(model=model, seq_len=config.trainer.seq_len, world_size=world_size)
        device_memory_monitor = build_device_memory_monitor()
        _ = utils.get_peak_flops(device_memory_monitor.device_name)
        _, _ = calculate_model_flops(model, config.trainer.seq_len)
        _ = device_memory_monitor.get_peak_stats()
        #     metric_logger = build_metrics_processor(job_config, parallel_dims)
        #     metric_logger.num_flops_per_token = num_flops_per_token
        device_memory_monitor.reset_peak_stats()

        start_step = 0
        epoch = 0

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

        # with (
        #     maybe_enable_profiling(
        #         job_config, global_step=train_state.step
        #     ) as torch_profiler,
        #     maybe_enable_memory_snapshot(
        #         job_config, global_step=train_state.step
        #     ) as memory_profiler,
        # ):

        for step in range(start_step, total_steps):
            if step % dataset.length == 0:
                epoch += 1
                dataset.set_epoch(epoch)

            gc_handler.run(step)
            optimizer.zero_grad()

            data_load_start = time.perf_counter()
            batch = next(data_iter)
            data_load_time = time.perf_counter() - data_load_start

            # metric_logger.ntokens_since_last_log += labels.numel()
            # metric_logger.data_loading_times.append(
            #     time.perf_counter() - data_load_start
            # )

            if config.model.compile_mode in ["reduce-overhead", "max-autotune"]:
                torch.compiler.cudagraph_mark_step_begin()

            labels = batch["input_ids"].clone()
            labels[batch["position_ids"] == 0] = -100  # mask document boundaries

            if "assistant_masks" in batch:
                labels[batch["assistant_masks"] == 0] = -100  # mask non-assistant tokens

            batch["labels"] = labels
            model_inputs = {k: v.cuda() for k, v in batch.items()}  # TODO: make the inputs explicit

            with autocast("cuda", dtype=torch.bfloat16):
                outputs = model(**model_inputs)
                loss = outputs.loss

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.max_norm)
            if hasattr(grad_norm, "full_tensor"):
                grad_norm = grad_norm.full_tensor()

            # grad_norm = dist_utils.clip_grad_norm_(
            #     [p for m in model_parts for p in m.parameters()],
            #     job_config.training.max_norm,
            # )

            # checkpoint.maybe_wait_for_staging()
            # if job_config.training.skip_nan_inf and (
            #     grad_norm.isnan() or grad_norm.isinf()
            # ):
            #     logger.warning(
            #         f"Skipping optimizer step - detected invalid gradient norm: {grad_norm:.4f}"
            #     )
            #     optimizers.zero_grad()
            #     train_state.skipped_step += 1

            optimizer.step()
            scheduler.step()

            redline.update(batch["input_ids"].numel(), data_load_time)

            if step % config.metrics.log_every == 0:
                current_lr = scheduler.get_last_lr()[0]
                current_loss = loss.item()
                # loss = loss.detach()
                metrics = redline.read()

                log_training_metrics(step, current_loss, grad_norm, current_lr, metrics, run_dir)
                log_wandb_metrics(step, current_loss, grad_norm, current_lr, metrics, wandb_run)

                # time_now = time.perf_counter()
                # time_delta = (
                #     time_now - metric_logger.time_last_log
                # )  # Use metric_logger's time
                # train_state.token += (
                #     metric_logger.ntokens_since_last_log  # Use tokens tracked by metric_logger
                #     * parallel_dims.world_size
                #     / parallel_dims.non_data_parallel_size
                # )
                # train_state.elapsed += timedelta(seconds=time_delta)
                # train_state.log_steps.append(train_state.step)
                # train_state.global_avg_losses.append(global_avg_loss)
                # train_state.global_max_losses.append(global_max_loss)

                # # Log using the metric processor
                # last_lr = lr_schedulers.schedulers[0].get_last_lr()[0]
                # eta = (
                #     train_state.elapsed
                #     * (job_config.training.steps - train_state.step)
                #     / train_state.step
                # )
                # metric_logger.log(
                #     train_state.step,
                #     global_avg_loss,
                #     global_max_loss,
                #     extra_metrics={
                #         "optimizer/lr": last_lr,
                #         "optimizer/grad_norm": grad_norm.item(),
                #         "optimizer/skipped_step": train_state.skipped_step,
                #     },
                # )

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

            # if torch_profiler:
            #     torch_profiler.step()
            # if memory_profiler:
            #     memory_profiler.step()

            # if train_state.step == 1:
            #     dist_utils.set_pg_timeouts(
            #         timeout=timedelta(seconds=job_config.comm.train_timeout_seconds),
            #         world_mesh=world_mesh,
            #     )

        save_checkpoint(
            step=redline.step,
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
        # if torch.distributed.get_rank() == 0:
        #     logger.info("Sleeping 2 seconds for other ranks to complete")
        #     time.sleep(2)
        destroy_dist()
        finish(wandb_run)
        cleanup_env()
