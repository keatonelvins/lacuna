"""Pydantic configs for pretraining and SFT."""

import shutil
from pathlib import Path
from typing import Literal, Optional

import torch
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelConfig(BaseModel):
    """Model config"""

    name: str = Field("Qwen/Qwen2.5-0.5B", description="HuggingFace model name or path")
    attention: Literal["FA3", "SDPA", "EAGER"] = Field(
        "FA3",
        description="Attention implementation (use FA3/SDPA, eager is just for baseline)",
    )
    accum_fp32: bool = Field(True, description="Use fp32 accumulation for cross entropy loss")
    liger: bool = Field(False, description="Enable Liger kernels")
    kernelize: bool = Field(False, description="Enable Hugging Face kernels.kernelize(model)")
    compile_mode: Optional[
        Literal[
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ]
    ] = Field(None, description="Compile mode (if omitted, torch.compile will not be applied)")


class OptimizerConfig(BaseModel):
    """Optimizer config"""

    type: Literal["adamw"] = "adamw"
    lr: float = Field(3e-4, gt=0, description="Learning rate")
    weight_decay: float = Field(0.01, ge=0, description="Weight decay")
    betas: tuple[float, float] = Field((0.9, 0.95), description="Adam betas")
    grad_clip: float = Field(1.0, gt=0, description="Gradient clipping norm")


class CosineSchedulerConfig(BaseModel):
    """Cosine scheduler config"""

    type: Literal["cosine"] = "cosine"
    warmup_ratio: float = Field(0.05, ge=0, le=1, description="Warmup ratio")
    min_lr_ratio: float = Field(0, ge=0, le=1, description="Minimum LR as ratio of max LR")


class WSDSchedulerConfig(BaseModel):
    """WSD scheduler config"""

    type: Literal["wsd"] = "wsd"
    warmup_steps: int = Field(100, ge=0, description="Warmup steps")
    decay_steps: int = Field(100, ge=0, description="Decay steps")
    min_lr_ratio: float = Field(0, ge=0, le=1, description="Minimum LR as ratio of max LR")
    decay_type: Literal["linear", "cosine"] = "linear"


class DataConfig(BaseModel):
    """Base config for data loading"""

    datasets: list[str] = Field(description="HF dataset names")
    split: str = Field("train", description="Split for all datasets")
    seq_len: int = Field(512, ge=1, description="Sequence length")
    stream: bool = Field(False, description="Stream in the datasets")
    num_workers: int = Field(1, ge=0, description="Number of workers for data loading")
    sampling_probs: list[float] = Field(default=None, description="Sampling probabilities for each dataset")


class PretrainDataConfig(DataConfig):
    """Pretraining data config"""

    datasets: list[str] = Field(["keatone/TinierStories"], description="HF dataset names")


class SFTDataConfig(DataConfig):
    """SFT data config"""

    datasets: list[str] = Field(["keatone/s1K"], description="HF dataset names")


class DistributedConfig(BaseModel):
    """Distributed training config"""

    backend: Literal["FSDP", "DDP"] = Field(
        "FSDP",
        description="FSDP for large models, DDP for small models",
    )
    cpu_offload: bool = Field(False, description="Offload params to CPU (enable if OOM, FSDP only)")
    hsdp: bool = Field(False, description="Enable HSDP (2D mesh: inter-node DDP + intra-node FSDP)")


class TorchrunConfig(BaseModel):
    """Torchrun distributed config"""

    nproc_per_node: int = Field(
        default_factory=lambda: torch.cuda.device_count(), ge=1, description="Number of processes per node"
    )
    nnodes: int = Field(1, ge=1, description="Number of nodes")
    master_addr: str = Field("localhost", description="Master node address")
    master_port: str = Field("29500", description="Master node port")
    node_rank: int = Field(None, ge=0, description="Node rank for multi-node training")


class TrainerConfig(BaseModel):
    """Training loop config"""

    batch_size: int = Field(1, ge=1, description="Global training batch size")


class MetricsConfig(BaseModel):
    """Metrics and logging config"""

    log_every: int = Field(10, gt=0, description="Steps between log outputs")


class WandbConfig(BaseModel):
    """Weights and Biases logging config"""

    enabled: bool = Field(False, description="Enable wandb logging")
    project: str = Field(None, description="wandb project name")
    name: str = Field(None, description="Run name")
    entity: str = Field(None, description="wandb entity (team/user)")
    offline: bool = Field(False, description="Run wandb in offline mode")


class ActivationCheckpointConfig(BaseModel):
    """Activation checkpointing config"""

    mode: Literal["none", "full", "partial"] = Field("none", description="Activation checkpointing mode")
    stride: int = Field(2, ge=1, description="If partial, checkpoint every nth layer")


class CheckpointConfig(BaseModel):
    """Checkpoint saving config"""

    save_every: int = Field(None, description="Steps between checkpoint saves (default no checkpointing)")
    save_dir: Path = Field(Path("weights"), description="Directory to save checkpoints")
    resume_from: Optional[Path] = Field(None, description="Checkpoint path to resume from")
    resumable_final_save: bool = Field(False, description="Make the final save resumable by storing optimizer state")

    def prepare_save_dir(self) -> None:
        """Clear save_dir if not resuming from checkpoint."""
        if not self.resume_from and self.save_dir.exists():
            shutil.rmtree(self.save_dir, ignore_errors=True)


class PretrainTrainerConfig(TrainerConfig):
    """Pretraining trainer config"""

    steps: int = Field(10000, gt=0, description="Maximum training steps")
    eval_every: int = Field(1000, gt=0, description="Steps between evaluations")


class SFTTrainerConfig(TrainerConfig):
    """SFT trainer config"""

    epochs: int = Field(3, gt=0, description="Number of epochs")
    eval_every: int = Field(1, gt=0, description="Epochs between evaluations")


class LacunaConfig(BaseSettings):
    """Shared base loader for pretraining and SFT"""

    model: ModelConfig = ModelConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    metrics: MetricsConfig = MetricsConfig()
    wandb: WandbConfig = WandbConfig()
    ac: ActivationCheckpointConfig = ActivationCheckpointConfig()
    dist: DistributedConfig = DistributedConfig()
    torchrun: TorchrunConfig = TorchrunConfig()


class PretrainConfig(LacunaConfig):
    """Pretraining config loader"""

    data: PretrainDataConfig = PretrainDataConfig()
    trainer: PretrainTrainerConfig = PretrainTrainerConfig()
    scheduler: WSDSchedulerConfig = WSDSchedulerConfig()

    model_config = SettingsConfigDict(
        cli_parse_args=True,
        cli_kebab_case=True,
        cli_implicit_flags=True,
    )


class SFTConfig(LacunaConfig):
    """SFT config loader"""

    data: SFTDataConfig = SFTDataConfig()
    trainer: SFTTrainerConfig = SFTTrainerConfig()
    scheduler: CosineSchedulerConfig = CosineSchedulerConfig()

    model_config = SettingsConfigDict(
        cli_parse_args=True,
        cli_kebab_case=True,
        cli_implicit_flags=True,
    )

