"""Pydantic configurations for pretraining and SFT."""

from enum import Enum
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelConfig(BaseModel):
    """Model config"""

    name: str = Field("Qwen/Qwen2.5-0.5B", description="HuggingFace model name or path")


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
    min_lr_ratio: float = Field(
        0, ge=0, le=1, description="Minimum LR as ratio of max LR"
    )


class WSDSchedulerConfig(BaseModel):
    """WSD scheduler config"""

    type: Literal["wsd"] = "wsd"
    warmup_steps: int = Field(100, ge=0, description="Warmup steps")
    decay_steps: int = Field(100, ge=0, description="Decay steps")
    min_lr_ratio: float = Field(
        0, ge=0, le=1, description="Minimum LR as ratio of max LR"
    )
    decay_type: Literal["linear", "cosine"] = "linear"


class DataConfig(BaseModel):
    """Base config for data loading"""

    dataset_name: str = Field(description="HF dataset name")
    split: str = Field("train", description="Dataset split")
    seq_len: int = Field(2048, ge=1, description="Sequence length")


class PretrainDataConfig(DataConfig):
    """Pretraining data config"""

    dataset_name: str = Field("keatone/TinierStories", description="HF dataset name")


class SFTDataConfig(DataConfig):
    """SFT data config"""

    dataset_name: str = Field("keatone/s1K", description="HF dataset name")
    packing: bool = Field(True, description="Pack multiple message lists per sample")


class FSDPShardingStrategy(str, Enum):
    """FSDP sharding strategies."""

    FULL_SHARD = "FULL_SHARD"
    SHARD_GRAD_OP = "SHARD_GRAD_OP"
    NO_SHARD = "NO_SHARD"
    HYBRID_SHARD = "HYBRID_SHARD"
    _HYBRID_SHARD_ZERO2 = "_HYBRID_SHARD_ZERO2"


class FSDPConfig(BaseModel):
    """FSDP distributed training config"""

    enabled: bool = Field(True, description="Enable FSDP (auto-detects multi-GPU)")
    reshard_after_forward: bool = Field(
        True, description="Reshard params after forward pass"
    )
    cpu_offload: bool = Field(False, description="Offload params to CPU for memory")
    sharding_strategy: FSDPShardingStrategy = Field(
        FSDPShardingStrategy.FULL_SHARD, description="FSDP sharding strategy"
    )


class TorchrunConfig(BaseModel):
    """Torchrun distributed config"""

    nproc_per_node: int = Field(8, ge=1, description="Number of processes per node")
    nnodes: int = Field(1, ge=1, description="Number of nodes")
    master_addr: str = Field("localhost", description="Master node address")
    master_port: str = Field("29500", description="Master node port")


class TrainerConfig(BaseModel):
    """Training loop config"""

    batch_size: int = Field(8, ge=1, description="Global training batch size")


class MetricsConfig(BaseModel):
    """Metrics and logging config"""

    log_every: int = Field(10, gt=0, description="Steps between log outputs")


class WandbConfig(BaseModel):
    """Weights and Biases logging config"""

    enabled: bool = Field(False, description="Enable wandb logging")
    project: str = Field("lacuna", description="Wandb project name")
    name: str = Field(None, description="Run name")
    entity: str = Field(None, description="Wandb entity (team/user)")
    offline: bool = Field(False, description="Run wandb in offline mode")


class CheckpointConfig(BaseModel):
    """Checkpoint saving config"""

    save_every: int = Field(1000, gt=0, description="Steps between checkpoint saves")
    keep_latest: int = Field(
        3, gt=0, description="Number of recent checkpoints to keep"
    )
    save_dir: Path = Field(Path("weights"), description="Directory to save checkpoints")
    resume_path: Optional[Path] = Field(
        None, description="Path to checkpoint to resume from"
    )


class PretrainTrainerConfig(TrainerConfig):
    """Pretraining trainer config"""

    steps: int = Field(10000, gt=0, description="Maximum training steps")
    eval_every: int = Field(1000, gt=0, description="Steps between evaluations")


class SFTTrainerConfig(TrainerConfig):
    """SFT trainer config"""

    epochs: int = Field(3, gt=0, description="Number of epochs")
    eval_every: int = Field(1, gt=0, description="Epochs between evaluations")


class PretrainConfig(BaseSettings):
    """Pretraining config loader"""

    model: ModelConfig = ModelConfig()
    data: PretrainDataConfig = PretrainDataConfig()
    trainer: PretrainTrainerConfig = PretrainTrainerConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    scheduler: WSDSchedulerConfig = WSDSchedulerConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    metrics: MetricsConfig = MetricsConfig()
    wandb: WandbConfig = WandbConfig()
    fsdp: FSDPConfig = FSDPConfig()
    torchrun: TorchrunConfig = TorchrunConfig()

    model_config = SettingsConfigDict(
        cli_parse_args=True,
        cli_kebab_case=True,
        cli_implicit_flags=True,
    )


class SFTConfig(BaseSettings):
    """SFT config loader"""

    model: ModelConfig = ModelConfig()
    data: SFTDataConfig = SFTDataConfig()
    trainer: SFTTrainerConfig = SFTTrainerConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    scheduler: CosineSchedulerConfig = CosineSchedulerConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    metrics: MetricsConfig = MetricsConfig()
    wandb: WandbConfig = WandbConfig()
    fsdp: FSDPConfig = FSDPConfig()
    torchrun: TorchrunConfig = TorchrunConfig()

    model_config = SettingsConfigDict(
        cli_parse_args=True,
        cli_kebab_case=True,
        cli_implicit_flags=True,
    )
