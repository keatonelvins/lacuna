"""Pydantic settings for jobs."""

import os
import wandb
import torch
import shutil
from pathlib import Path
from typing import Literal, Optional
from pydantic_settings import BaseSettings
from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelConfig(StrictModel):
    """Modeling and patching config"""

    name: str = Field("Qwen/Qwen3-0.6B-Base", description="HuggingFace model name or (local?) path")
Ac    backend: Literal["hf", "liger", "lacuna"] = Field("liger", description="Modeling backend")
    attention: str = Field("kernels-community/flash-attn3", description="Attention backend")
    kernelize: bool = Field(False, description="Enable Hugging Face kernels.kernelize(model)")
    compile_mode: Literal["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"] = Field(
        None, description="Compile mode (if omitted, torch.compile will not be used)"
    )


class TrainerConfig(StrictModel):
    """Training loop config"""

    seed: int = Field(42, description="Global seed")
    seq_len: int = Field(512, ge=1, description="Tokens per GPU (batch size is always 1)")
    epochs: int = Field(1, gt=0, description="Number of epochs")
    steps: int = Field(None, gt=0, description="Max training steps (overrides epochs, must be < dataset.length)")


class DatasetConfig(StrictModel):
    """Dataset config (matching datasets.load_dataset API)"""

    path: str = Field("keatone/TinierStories", description="Name or path or builder type of the dataset")
    name: str = Field(None, description="Name of the dataset configuration")
    data_dir: str = Field(None, description="Data directory of the dataset configuration")
    data_files: str | list[str] | dict[str, str] = Field(None, description="Data files of the dataset configuration")
    split: str = Field("train", description="Split to use")


class DataConfig(StrictModel):
    """Data loading config"""

    datasets: list[DatasetConfig] = Field([DatasetConfig()], description="Datasets to use")
    column: str = Field("text", description="Column to use for all datasets")
    context_len: int = Field(None, ge=1, description="Max length of a single example (defaults to seq_len)")
    truncate: bool = Field(True, description="Whether to truncate long examples to context_len (above) or just drop them")
    chat_template: str = Field(None, description="Chat template to use (either a string or a path to a file)")
    eos_token: str = Field(None, description="New eos token (required if adding a chat template to a base model)")
    override_tokenizer: str = Field(None, description="Model name to override the default tokenizer (for caching purposes)")
    override_cache: bool = Field(False, description="Force redownload of the dataset to avoid cache reuse")
    tok_bs: int = Field(10000, description="Batch size to use when tokenizing the dataset")
    tok_num_proc: int = Field(os.cpu_count(), description="Number of processes to use while tokenizing")
    pack_bs: int = Field(100000, description="Batch size to use when packing the dataset")
    pack_num_proc: int = Field(None, description="Number of processes to use while packing (defaults to dataset.length / pack_bs)")
    num_workers: int = Field(1, description="Number of workers to use for the torch DataLoader")

    @field_validator("chat_template", mode="after")
    @classmethod
    def validate_chat_template(cls, chat_template: str | None) -> str:
        """If chat_template is a file path, read the template from the file."""
        if chat_template and chat_template.endswith(".jinja"):
            maybe_template_path = Path(chat_template)
            if maybe_template_path.exists() and maybe_template_path.is_file():
                return maybe_template_path.read_text()
        return chat_template


class EnvConfig(StrictModel):
    """Env config"""

    name: str = Field(description="Env name")


class EvalsConfig(StrictModel):
    """Evals config"""

    datasets: list[DatasetConfig] = Field(default_factory=list, description="Datasets to use for eval")
    envs: list[EnvConfig] = Field(default_factory=list, description="vf environments to use for eval")


class OptimizerConfig(StrictModel):
    """Optimizer config"""

    name: Literal["adamw"] = Field("adamw", description="Optimizer name (adamw only for now)")
    lr: float = Field(3e-4, gt=0, description="Peak learning rate")
    weight_decay: float = Field(0.1, ge=0, description="Weight decay (not applied to embeddings etc.)")
    betas: tuple[float, float] = Field((0.9, 0.95), description="Adam betas")
    eps: float = Field(1e-8, ge=0, description="Adam eps")
    max_norm: float = Field(1.0, gt=0, description="Gradient clipping norm ")


class SchedulerConfig(StrictModel):
    """LR Scheduler config"""

    warmup_ratio: float = Field(0.05, ge=0, le=1, description="Warmup ratio over total steps")
    decay_ratio: float = Field(0.20, ge=0, le=1, description="Decay ratio over total steps")
    min_lr_ratio: float = Field(0, ge=0, le=1, description="Minimum LR as ratio of max LR")
    decay_type: Literal["linear", "cosine"] = "linear"


class CheckpointConfig(StrictModel):
    """Checkpoint saving config"""

    save_every: int = Field(None, gt=0, description="Steps between checkpoint saves (default no checkpointing)")
    save_dir: Optional[Path] = Field(None, description="Directory to save checkpoints to")
    resume_from: Optional[Path] = Field(None, description="Checkpoint path to resume from")

    def prepare_save_dir(self, timestamp: str) -> None:
        """Clear save_dir if not resuming from checkpoint."""
        self.save_dir = self.save_dir or Path("weights") / timestamp

        if not self.resume_from and self.save_dir.exists():
            shutil.rmtree(self.save_dir, ignore_errors=True)


class MetricsConfig(StrictModel):
    """Metrics and logging config"""

    log_every: int = Field(1, gt=0, description="Steps between log outputs")


class TorchrunConfig(StrictModel):
    """Torchrun distributed config (matching torchrun API)"""

    nproc_per_node: int = Field(
        default_factory=lambda: torch.cuda.device_count(), ge=1, description="Number of processes per node"
    )
    nnodes: int = Field(1, ge=1, description="Number of nodes")
    master_addr: str = Field("localhost", description="Master node address")
    master_port: str = Field("29500", description="Master node port")
    node_rank: int = Field(None, ge=0, description="Node rank for multi-node training")


class DistributedConfig(StrictModel):
    """Distributed training and parallelism config"""

    dp_replicate: int = Field(None, ge=1, description="Number of replicated DP groups (defaults to nnodes)")
    dp_shard: int = Field(None, ge=1, description="Number of shards per replica (defaults to nproc_per_node)")
    cpu_offload: bool = Field(False, description="Offload params to CPU (enable if OOM, FSDP only)")
    reshard_after_forward: bool = Field(None, description="Reshard after forward pass (default is True except for root module)")


class ActivationCheckpointConfig(StrictModel):
    """Activation checkpointing config"""

    stride: int = Field(0, ge=0, description="Checkpoint every nth layer (0 means no checkpointing)")


class WandbConfig(StrictModel):
    """Weights and Biases logging config"""

    project: str = Field(None, description="wandb project name")
    name: str = Field(None, description="wandb run name")
    entity: str = Field(None, description="wandb entity (team/user)")
    offline: bool = Field(False, description="Enable running wandb in offline mode")

    @field_validator("project", mode="after")
    @classmethod
    def validate_wandb_login(cls, project: str) -> None:
        if project and not wandb.api.api_key:
            raise RuntimeError("wandb project specified but no api key found, please login first.")
        return project


class LacunaConfig(BaseSettings):
    """Base loader for training configs"""

    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    evals: EvalsConfig = EvalsConfig()
    trainer: TrainerConfig = TrainerConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    metrics: MetricsConfig = MetricsConfig()
    torchrun: TorchrunConfig = TorchrunConfig()
    dist: DistributedConfig = DistributedConfig()
    ac: ActivationCheckpointConfig = ActivationCheckpointConfig()
    wandb: WandbConfig = WandbConfig()
