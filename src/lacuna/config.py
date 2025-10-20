"""Pydantic settings for training jobs."""

import os
import torch
from pathlib import Path
from typing import Literal
from datetime import datetime
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, model_validator, BaseModel, ConfigDict


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelConfig(StrictModel):
    """Modeling and patching config"""

    name: str = Field("Qwen/Qwen3-0.6B-Base", description="hf model name or local path")
    attn: str = Field("kernels-community/flash-attn3", description="Attention backend")
    lacuna: bool = Field(True, description="Lacunaify model (liger + tt moe), otherwise use vanilla hf backend")
    fp8: bool = Field(False, description="Enable FP8 training (tensorwise with FSDP tricks from tt)")


class CompileConfig(StrictModel):
    """torch.compile config"""

    mode: str | None = Field(None, description="Compile mode e.g. 'default', 'max-autotune-no-cudagraphs'")
    fullgraph: bool = Field(False, description="Force compile to single graph")
    dynamic: bool | None = Field(None, description="Use dynamic shape tracing")
    compile_optimizer_step: bool = Field(False, description="Compile optimizer.step()")


class TrainerConfig(StrictModel):
    """Training loop config"""

    seed: int = Field(42, description="Global seed")
    steps: int | None = Field(None, gt=0, description="Max training steps (overrides epochs, must be < len(dataset))")
    epochs: int = Field(1, gt=0, description="Number of epochs")
    seq_len: int = Field(512, ge=1, description="Tokens per GPU")
    pad_to: int = Field(1, ge=1, description="Pad sequences to multiple of this value")
    run_name: str | None = Field(None, description="Name of experiment")


class DatasetConfig(StrictModel):
    """Dataset config (matching hf datasets.load_dataset API)"""

    path: str = Field("keatone/TinierStories", description="Name or path or builder type of the dataset")
    split: str = Field("train", description="Split to use")
    name: str | None = Field(None, description="Name of the dataset config")
    data_dir: str | None = Field(None, description="Data directory of the dataset config")
    data_files: str | list[str] | dict[str, str] | None = Field(None, description="Data files of the dataset config")


class DataConfig(StrictModel):
    """Data loading config"""

    datasets: list[DatasetConfig] = Field(default_factory=lambda: [DatasetConfig()], description="Datasets to train on")
    column: str = Field("text", description="Column to tokenize ('text' or 'messages' most common)")
    context_len: int | None = Field(None, ge=1, description="Max length of a single example (defaults to seq_len)")
    truncate: bool = Field(False, description="Whether to truncate examples to context_len or instead drop them")
    chat_template: str | None = Field(None, description="Chat template to use (either a string or a path to a file)")
    eos_token: str | None = Field(None, description="New eos token (for SFT on base model)")
    skip_cache: bool = Field(False, description="Skip cache and force redownload of the dataset")
    tok_bs: int = Field(10000, description="Batch size to use when tokenizing the dataset")
    tok_num_proc: int = Field(os.cpu_count(), description="Number of processes to use while tokenizing")
    pack_bs: int = Field(100000, description="Batch size to use when packing the dataset")
    pack_num_proc: int | None = Field(None, description="Packing process count (defaults to len(dataset) / pack_bs)")
    num_workers: int = Field(1, description="Number of workers to use for the torch DataLoader")


class EvalsConfig(StrictModel):
    """Evals config"""

    datasets: list[DatasetConfig] = Field(default_factory=list, description="Datasets to run eval on")


class OptimizerConfig(StrictModel):
    """Optimizer config"""

    name: Literal["adamw", "adamw_8bit", "muon"] = Field("adamw", description="Optimizer name")
    weight_decay: float = Field(0.1, ge=0, description="Weight decay")
    betas: tuple[float, float] = Field((0.9, 0.95), description="Adam betas")
    eps: float = Field(1e-8, ge=0, description="Adam eps")
    max_norm: float = Field(1.0, gt=0, description="Gradient clipping norm ")


class SchedulerConfig(StrictModel):
    """LR Scheduler config"""

    lr: float = Field(3e-4, gt=0, description="Peak learning rate")
    start_lr: float = Field(1e-8, gt=0, description="Starting learning rate for warmup")
    end_lr: float = Field(0, ge=0, description="Minimum learning rate after decay")
    warmup_ratio: float = Field(0.05, ge=0, le=1, description="Warmup ratio to total steps")
    decay_ratio: float = Field(0.20, ge=0, le=1, description="Decay ratio to total steps")
    decay_type: Literal["linear", "cosine"] = Field("linear", description="Shape of decay")


class CheckpointConfig(StrictModel):
    """Checkpoint saving config"""

    save_dir: Path | None = Field(None, description="Checkpoint dir (defaults to weights/trainer.run_name)")
    saves: int = Field(0, ge=0, description="Total number of intermediate checkpoints to save.")
    save_every: int | None = Field(None, gt=0, description="If saves > 0, force steps between checkpoint saves")
    resumable: bool = Field(False, description="Whether to save resumable dcp checkpoints or just hf safetensors")
    resume_from: Path | None = Field(None, description="Checkpoint path to resume from (must be dcp currently)")
    full_state: bool = Field(True, description="Whether to load the full state of the checkpoint or just model/optim")


class MetricsConfig(StrictModel):
    """Metrics and logging config"""

    log_every: int = Field(1, gt=0, description="Steps between log outputs")


class ProfileConfig(StrictModel):
    """torch profiler and memory snapshots config"""

    enable_profiling: bool = False
    enable_memory_snapshot: bool = False
    profile_freq: int = 100


class TorchrunConfig(StrictModel):
    """Torchrun distributed config (matching torchrun API)"""

    job_id: str = Field("101", description="Job ID for multi-node training")
    nnodes: int = Field(1, ge=1, description="Number of nodes")
    nproc_per_node: int = Field(torch.cuda.device_count(), ge=1, description="Number of processes per node")
    master_addr: str = Field("localhost", description="Master node address")
    master_port: str = Field("29500", description="Master node port")
    node_rank: int | None = Field(None, ge=0, description="Node rank for multi-node training")


class DistributedConfig(StrictModel):
    """Distributed training and parallelism config"""

    dp_replicate: int | None = Field(None, ge=1, description="Number of replicated DP groups (default nnodes)")
    dp_shard: int | None = Field(None, ge=1, description="Number of shards per replica (default nproc_per_node)")
    reshard: bool | None = Field(
        None, description="Reshard after forward pass (default is True except for root module)"
    )


class ActivationCheckpointConfig(StrictModel):
    """Activation checkpointing config"""

    stride: int = Field(0, ge=0, description="Checkpoint every nth layer (0 means no checkpointing)")


class WandbConfig(StrictModel):
    """Weights and Biases logging config"""

    project: str | None = Field(None, description="wandb project name")
    name: str | None = Field(None, description="wandb run name")
    entity: str | None = Field(None, description="wandb entity (team/user)")
    offline: bool = Field(False, description="Enable running wandb in offline mode")


class SlurmConfig(StrictModel):
    """slurm cluster config"""

    duration: int = Field(168, ge=1, description="Max job time in hours")
    queue: bool = Field(False, description="Reuse nodes from queued/running jobs instead of spinning up more.")


class TrainConfig(BaseSettings):
    """Base loader for training configs"""

    model_config = SettingsConfigDict(env_prefix="LACUNA", env_nested_delimiter="__", extra="forbid")

    model: ModelConfig = ModelConfig()
    compile: CompileConfig = CompileConfig()
    data: DataConfig = DataConfig()
    evals: EvalsConfig = EvalsConfig()
    trainer: TrainerConfig = TrainerConfig()
    optim: OptimizerConfig = OptimizerConfig()
    sched: SchedulerConfig = SchedulerConfig()
    ckpt: CheckpointConfig = CheckpointConfig()
    metrics: MetricsConfig = MetricsConfig()
    profile: ProfileConfig = ProfileConfig()
    torchrun: TorchrunConfig = TorchrunConfig()
    dist: DistributedConfig = DistributedConfig()
    ac: ActivationCheckpointConfig = ActivationCheckpointConfig()
    wandb: WandbConfig = WandbConfig()
    slurm: SlurmConfig = SlurmConfig()

    @model_validator(mode="after")
    def setup_chat_template(self):
        if self.data.chat_template and self.data.chat_template.endswith(".jinja"):
            maybe_template_path = Path(self.data.chat_template)
            if maybe_template_path.exists() and maybe_template_path.is_file():
                self.data.chat_template = maybe_template_path.read_text()

        return self

    @model_validator(mode="after")
    def setup_dp_config(self):
        if self.dist.dp_replicate is None:
            self.dist.dp_replicate = self.torchrun.nnodes
        if self.dist.dp_shard is None:
            self.dist.dp_shard = self.torchrun.nproc_per_node
        return self

    @model_validator(mode="after")
    def setup_experiment(self):
        if self.wandb.name and not self.trainer.run_name:
            self.trainer.run_name = self.wandb.name + "_" + datetime.now().strftime("%H%M%S")
        elif not self.trainer.run_name:
            self.trainer.run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        if not self.ckpt.save_dir:
            self.ckpt.save_dir = Path("weights") / self.trainer.run_name

        return self

    @model_validator(mode="after")
    def validate_fp8_compile(self):
        if self.model.fp8 and self.compile.mode is None:
            raise ValueError("FP8 training requires compile.mode to be set")
        return self


class SweepConfig:
    """Run consecutive training jobs, sweeping over specified parameters.

    Usage:
        uv run sweep [config.toml] --param1 val1,val2,val3 --param2 start:stop:step --fixed_param value

    Examples:
        uv run sweep --sched.lr 1e-4,3e-4,1e-3 --optim.weight_decay 0.0,0.1
        uv run sweep configs/base.toml --sched.lr 1e-5:1e-3:1e-5 --trainer.epochs 3

    Sweep syntax:
        - Comma-separated: --param a,b,c  (discrete values)
        - Range notation: --param start:stop:step  (generates range)
        - Fixed values: everything from .toml and instances of --param value (no comma or colon)
    """
