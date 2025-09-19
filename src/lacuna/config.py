"""Pydantic settings."""

import shutil
from pathlib import Path
from typing import Literal, Optional

import torch
import psutil
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelConfig(BaseModel):
    """Modeling and patching config"""

    name: str = Field("Qwen/Qwen2.5-0.5B", description="HuggingFace model name or path")
    attention: Literal["FA3"] = Field("FA3", description="Attention implementation (FA3 only for now)")
    accum_fp32: bool = Field(True, description="Enable fp32 accumulation for cross entropy loss")
    liger: bool = Field(False, description="Enable Liger kernels")
    kernelize: bool = Field(False, description="Enable Hugging Face kernels.kernelize(model)")
    compile_mode: Optional[Literal["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"]] = Field(
        None, description="Compile mode (if omitted, torch.compile will not be used)"
    )


class TrainerConfig(BaseModel):
    """Training loop config"""

    seed: int = Field(42, description="Global seed")
    seq_len: int = Field(512, ge=1, description="Tokens per GPU (batch size is always 1)")
    epochs: int = Field(1, gt=0, description="Number of epochs")
    steps: Optional[int] = Field(None, gt=0, description="Max training steps (must be less than dataset length)")


class DatasetConfig(BaseModel):
    """Dataset config (matches hf load_dataset API)"""

    path: str = Field("keatone/TinierStories", description="Path or name of the dataset")
    name: str = Field(None, description="Name of the dataset configuration")
    data_dir: str = Field(None, description="Data directory of the dataset configuration")
    data_files: str | list[str] | dict[str, str] = Field(None, description="Data files of the dataset configuration")
    split: str = Field("train", description="Split to use")


class DataConfig(BaseModel):
    """Data loading config"""

    datasets: list[DatasetConfig] = Field([DatasetConfig(path="keatone/TinierStories")], description="Datasets to use")
    column: str = Field("text", description="Column to use for all datasets")
    chat_template: str = Field(None, description="Chat template to use for the dataset (either a string or a path to a file)")
    eos_token: str = Field(None, description="New eos token (required if adding a chat template to a base model)")
    tokenizer_override: str = Field(None, description="Model name to override the default tokenizer")
    redownload: bool = Field(False, description="Force redownload of the dataset to avoid cache reuse")
    fingerprint: str = Field(None, description="Fingerprint of the dataset to use for caching")
    map_bs: int = Field(10000, description="Batch size to use when tokenizing the dataset")
    pack_bs: int = Field(10000, description="Batch size to use when packing the dataset")
    num_proc: int = Field(psutil.cpu_count(logical=False), description="Number of processes to use for dataset.map()")
    num_workers: int = Field(1, description="Number of workers to use for the torch DataLoader")

    @field_validator("chat_template", mode="after")
    @classmethod
    def validate_chat_template(cls, chat_template: str | None) -> str:
        """If chat_template is a file path, read the template from the file."""
        if chat_template:
            try:
                maybe_template_path = Path(chat_template)
                if maybe_template_path.exists() and maybe_template_path.is_file():
                    return maybe_template_path.read_text()
            except Exception:
                pass
        return chat_template


class OptimizerConfig(BaseModel):
    """Optimizer config"""

    type: Literal["adamw"] = Field("adamw", description="Optimizer type (adamw only for now)")
    lr: float = Field(3e-4, gt=0, description="Peak learning rate")
    weight_decay: float = Field(0.01, ge=0, description="Weight decay (not applied to embeddings etc.)")
    betas: tuple[float, float] = Field((0.9, 0.95), description="Adam betas")
    max_norm: float = Field(1.0, gt=0, description="Gradient clipping norm ")


class SchedulerConfig(BaseModel):
    """LR Scheduler config"""

    warmup_ratio: float = Field(0.05, ge=0, le=1, description="Warmup ratio over total steps")
    decay_ratio: float = Field(0.05, ge=0, le=1, description="Decay ratio over total steps")
    min_lr_ratio: float = Field(0, ge=0, le=1, description="Minimum LR as ratio of max LR")
    decay_type: Literal["linear", "cosine"] = "linear"


class CheckpointConfig(BaseModel):
    """Checkpoint saving config"""

    save_every: int = Field(None, gt=0, description="Steps between checkpoint saves (default no checkpointing)")
    save_dir: Path = Field(Path("weights"), description="Directory to save checkpoints")
    resume_from: Optional[Path] = Field(None, description="Checkpoint path to resume from")
    resumable_final_save: bool = Field(False, description="Make the final save resumable by storing optimizer state")

    def prepare_save_dir(self) -> None:
        """Clear save_dir if not resuming from checkpoint."""
        if not self.resume_from and self.save_dir.exists():
            shutil.rmtree(self.save_dir, ignore_errors=True)


class MetricsConfig(BaseModel):
    """Metrics and logging config"""

    log_every: int = Field(10, gt=0, description="Steps between log outputs")


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

    dp_replicate: int = Field(None, ge=1, description="Number of replicated DP groups (defaults to nnodes)")
    dp_shard: int = Field(None, ge=1, description="Number of shards per replica (defaults to nproc_per_node)")
    cpu_offload: bool = Field(False, description="Offload params to CPU (enable if OOM, FSDP only)")


class ActivationCheckpointConfig(BaseModel):
    """Activation checkpointing config"""

    mode: Literal["none", "full", "partial"] = Field("none", description="Activation checkpointing mode")
    stride: int = Field(2, ge=1, description="If partial, checkpoint every nth layer")


class WandbConfig(BaseModel):
    """Weights and Biases logging config"""

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
