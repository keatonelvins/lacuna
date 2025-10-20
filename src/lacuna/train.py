"""Core training loop."""

import time
import torch
from pathlib import Path
from loguru import logger
from torchtitan.tools import utils
from torchtitan.config import Profiling as tt_profiling_config
from torchtitan.distributed.utils import dist_mean, clip_grad_norm_ as clip
from torchtitan.tools.profiling import maybe_enable_profiling, maybe_enable_memory_snapshot

from lacuna.eval import run_eval
from lacuna.config import TrainConfig
from lacuna.data import PackedDataset
from lacuna.optim import build_optimizer
from lacuna.sched import build_scheduler
from lacuna.parallelize import apply_parallelisms
from lacuna.model import build_model, initialize_buffers
from lacuna.ckpt import save_checkpoint, load_checkpoint, load_pretrained_weights
from lacuna.monitor import init_wandb, log_wandb_metrics, finish
from lacuna.utils import (
    init_dist,
    destroy_dist,
    set_seed,
    log_train_metrics,
    log_eval_metrics,
    get_moe_stats,
    setup_run_dir,
    setup_logger,
    MetricsProcessor,
)


@logger.catch(reraise=True)
def train(config: TrainConfig) -> None:
    init_dist()
    set_seed(config.trainer.seed)
    run_dir = Path("runs") / config.trainer.run_name
    setup_run_dir(config, run_dir)
    setup_logger(run_dir)
    wandb_run = init_wandb(config)
    gc_handler = utils.GarbageCollection()

    model, fp8_converter = build_model(config)
    model, mesh, amp = apply_parallelisms(model, config)

    logger.info("Initializing model...")
    model.to_empty(device="cuda")

    logger.info("Building packed dataset")
    dataset = PackedDataset(config, mesh=mesh)
    logger.info(f"Packed dataset length: {dataset.length}")

    if config.trainer.steps:
        step, total_steps = 0, config.trainer.steps
    else:
        step, total_steps = 0, dataset.length * config.trainer.epochs
    logger.info(f"Total training steps: {total_steps}")

    if config.ckpt.saves > 0:
        if config.ckpt.save_every is None:
            config.ckpt.save_every = max(1, round(total_steps / (config.ckpt.saves + 1)))  # best effort
            logger.info(f"Total saves: {config.ckpt.saves}, saving every {config.ckpt.save_every} steps")
        else:
            logger.info(f"Total saves: {config.ckpt.saves}, saving every {config.ckpt.save_every} steps")

    optimizer = build_optimizer(model, config, mesh)
    scheduler = build_scheduler(optimizer, config.sched, total_steps)
    metrics_processor = MetricsProcessor(config)

    if fp8_converter is not None:
        fp8_converter.post_optimizer_hook(model)

    if config.compile.compile_optimizer_step:
        optimizer.step = torch.compile(optimizer.step, mode=config.compile.mode)

    if config.ckpt.resume_from:
        step = load_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler if config.ckpt.full_state else None,
            dataloader=dataset.dataloader if config.ckpt.full_state else None,
            path=config.ckpt.resume_from,
        )
        if config.ckpt.full_state:
            logger.info(f"Resumed from step {step} (full state)")
        else:
            logger.info(f"Loaded model+optimizer from step {step}, resetting to step 0")
            step = 0
    else:
        load_pretrained_weights(model, config.model.name)

    initialize_buffers(model)
    model.train()
    logger.info("Model initialized!")

    data_iter = iter(dataset.dataloader)
    logger.info(f"Starting training at step {step + 1}")

    profiling_config = tt_profiling_config(**config.profile.model_dump())
    torch_profiler = maybe_enable_profiling(profiling_config, global_step=step, base_folder=str(run_dir))
    memory_profiler = maybe_enable_memory_snapshot(profiling_config, global_step=step, base_folder=str(run_dir))

    with torch_profiler as prof, memory_profiler as mem_prof:
        while step < total_steps:
            step += 1
            epoch = step // dataset.length

            if step % dataset.length == 1:
                logger.info(f"Setting epoch {epoch}")
                dataset.set_epoch(epoch)
                data_iter = iter(dataset.dataloader)

            gc_handler.run(step)
            optimizer.zero_grad()

            data_load_start = time.perf_counter()
            batch = next(data_iter)
            loading_time = time.perf_counter() - data_load_start

            ntokens_batch = batch["input_ids"].numel()
            metrics_processor.data_loading_times.append(loading_time)
            metrics_processor.ntokens_since_last_log += ntokens_batch

            if config.compile.mode in ["reduce-overhead", "max-autotune"]:
                torch.compiler.cudagraph_mark_step_begin()

            model_inputs = {
                "input_ids": batch["input_ids"].cuda(),
                "labels": batch["labels"].cuda(),
                "position_ids": batch["position_ids"].cuda(),
                "accum_dtype": torch.float32,  # used by liger for flce
            }

            with amp:
                loss = model(**model_inputs).loss

            loss.backward()
            grad_norm = clip(model.parameters(), config.optim.max_norm)
            optimizer.step()
            scheduler.step()

            if prof:
                prof.step()
            if mem_prof:
                mem_prof.step()

            if step % config.metrics.log_every == 0:
                current_lr = scheduler.get_last_lr()[0]
                current_lr = current_lr.item() if isinstance(current_lr, torch.Tensor) else current_lr
                current_grad_norm = grad_norm.item()  # already reduced
                if mesh:
                    loss_mesh = mesh["dp"] if mesh.ndim > 1 else mesh
                    current_loss = dist_mean(loss.detach(), loss_mesh)
                else:
                    current_loss = loss.detach().item()

                metrics = {
                    "train/loss": current_loss,
                    "train/lr": current_lr,
                    "train/grad_norm": current_grad_norm,
                    "train/ntokens_micro_batch": ntokens_batch,
                    **metrics_processor.get_metrics(),
                    **get_moe_stats(model),
                }
                log_train_metrics(step, metrics, run_dir)
                log_wandb_metrics(step, metrics, wandb_run)

            if config.ckpt.saves > 0 and step % config.ckpt.save_every == 0:
                save_checkpoint(
                    step=step,
                    config=config,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    dataloader=dataset.dataloader,
                    resumable=config.ckpt.resumable,
                )

    save_checkpoint(
        step=step,
        config=config,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloader=dataset.dataloader,
        resumable=config.ckpt.resumable,
        final=True,
    )

    if config.evals.datasets:
        eval_metrics = run_eval(config, model, amp, mesh)
        log_eval_metrics(step, eval_metrics, run_dir)
        log_wandb_metrics(step, eval_metrics, wandb_run)

    destroy_dist()
    finish(wandb_run)
