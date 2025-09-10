"""Pydantic settings."""

import shutil
from pathlib import Path
from typing import Literal, Optional

import torch
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelConfig(BaseModel):
    """Modeling and patching config"""

    name: str = Field("Qwen/Qwen2.5-0.5B", description="HuggingFace model name or path")
    attention: Literal["FA3"] = Field("FA3", description="Attention implementation (FA3 only for now)")
    accum_fp32: bool = Field(True, description="Enable fp32 accumulation for cross entropy loss")
    liger: bool = Field(False, description="Enable Liger kernels")
    kernelize: bool = Field(False, description="Enable Hugging Face kernels.kernelize(model)")
    compile_mode: Optional[
        Literal[
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ]
    ] = Field(None, description="Compile mode (if omitted, torch.compile will not be used)")


class TrainerConfig(BaseModel):
    """Training loop config"""

    seed: int = Field(42, description="Global seed")
    epochs: int = Field(1, gt=0, description="Number of epochs")
    steps: Optional[int] = Field(None, gt=0, description="Max training steps (will override epochs if set)")
    seq_len: int = Field(512, ge=1, description="Sequence length")
    batch_size: int = Field(1, ge=1, description="Global training batch size across all GPUs")
    eval_every: float = Field(None, gt=0, le=1, description="Eval frequency in epochs, can be fractional (default no eval)")


class DataConfig(BaseModel):
    """Data loading config"""

    datasets: list[str] = Field(["keatone/TinierStories"], description="HF dataset names")
    files: dict = Field(default_factory=dict, description="Mapping of dataset to regex for fs matching")
    split: str = Field("train", description="Split to use for all the datasets")
    stream: bool = Field(False, description="Enable streaming the datasets (e.g. if it's too big to fit on disk)")
    sampling_probs: list[float] = Field(default=None, description="If streaming multiple datasets, how to sample from them")
    shuffle_buffer: int = Field(
        50000, description="If streaming, the num samples shuffled at a time (for disk we shuffle the entire dataset)"
    )
    map_batch_size: int = Field(10000, description="Batch size to use when tokenizing the dataset")
    pack_batch_size: int = Field(10000, description="Batch size to use when packing the dataset")
    num_workers: int = Field(4, description="The number of workers to use for data loading")

    @model_validator(mode="after")
    def set_files(self):
        for dataset in self.datasets:
            if dataset.startswith("s3://") and dataset not in self.files:
                self.files[dataset] = {self.split: dataset.rstrip("/") + f"/{self.split}/*.parquet"}
        return self


class OptimizerConfig(BaseModel):
    """Optimizer config"""

    type: Literal["adamw"] = Field("adamw", description="Optimizer type (adamw only for now)")
    lr: float = Field(3e-4, gt=0, description="Peak learning rate")
    weight_decay: float = Field(0.01, ge=0, description="Weight decay (not applied to embeddings etc.)")
    betas: tuple[float, float] = Field((0.9, 0.95), description="Adam betas")
    grad_clip: float = Field(1.0, gt=0, description="Gradient clipping norm ")


class SchedulerConfig(BaseModel):
    """LR Scheduler config"""

    warmup_ratio: float = Field(0.05, ge=0, le=1, description="Warmup ratio for the scheduler")
    decay_ratio: float = Field(0.05, ge=0, le=1, description="Decay ratio for the scheduler")
    min_lr_ratio: float = Field(0, ge=0, le=1, description="Minimum LR as ratio of max LR")
    decay_type: Literal["linear", "cosine"] = "linear"


class CheckpointConfig(BaseModel):
    """Checkpoint saving config"""

    save_every: float = Field(None, description="Epochs between checkpoint saves (default no checkpointing)")
    save_dir: Path = Field(Path("weights"), description="Directory to save checkpoints")
    resume_from: Optional[Path] = Field(None, description="Checkpoint path to resume from")
    resumable_final_save: bool = Field(False, description="Make the final save resumable by storing optimizer state")

    def prepare_save_dir(self) -> None:
        """Clear save_dir if not resuming from checkpoint."""
        if not self.resume_from and self.save_dir.exists():
            shutil.rmtree(self.save_dir, ignore_errors=True)


class MetricsConfig(BaseModel):
    """Metrics and logging config"""

    steps_per_log: int = Field(10, gt=0, description="Steps between log outputs")


class TorchrunConfig(BaseModel):
    """Torchrun distributed config"""

    nproc_per_node: int = Field(
        default_factory=lambda: torch.cuda.device_count(), ge=1, description="Number of processes per node"
    )
    nnodes: int = Field(1, ge=1, description="Number of nodes")
    master_addr: str = Field("localhost", description="Master node address")
    master_port: str = Field("29500", description="Master node port")
    node_rank: int = Field(None, ge=0, description="Node rank for multi-node training")


class DistributedConfig(BaseModel):
    """Distributed training config"""

    backend: Literal["FSDP", "DDP"] = Field(
        "FSDP",
        description="FSDP for large models, DDP for small models",
    )
    cpu_offload: bool = Field(False, description="Offload params to CPU (enable if OOM, FSDP only)")
    hsdp: bool = Field(False, description="Enable HSDP (2D mesh: inter-node DDP + intra-node FSDP)")


class ActivationCheckpointConfig(BaseModel):
    """Activation checkpointing config"""

    mode: Literal["none", "full", "partial"] = Field("none", description="Activation checkpointing mode")
    stride: int = Field(2, ge=1, description="If partial, checkpoint every nth layer")


class WandbConfig(BaseModel):
    """Weights and Biases logging config"""

    enabled: bool = Field(False, description="Enable wandb logging")
    project: str = Field(None, description="wandb project name")
    name: str = Field(None, description="wandb run name")
    entity: str = Field(None, description="wandb entity (team/user)")
    offline: bool = Field(False, description="Enable running wandb in offline mode")


class LacunaConfig(BaseSettings):
    """Base loader for training configs"""

    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    trainer: TrainerConfig = TrainerConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    metrics: MetricsConfig = MetricsConfig()
    torchrun: TorchrunConfig = TorchrunConfig()
    dist: DistributedConfig = DistributedConfig()
    ac: ActivationCheckpointConfig = ActivationCheckpointConfig()
    wandb: WandbConfig = WandbConfig()

    model_config = SettingsConfigDict(
        cli_parse_args=True,
        cli_kebab_case=True,
        cli_implicit_flags=True,
    )
