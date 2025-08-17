"""Pydantic configurations for pretraining and SFT."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelConfig(BaseModel):
    """Model loading config"""

    name: str = Field("Qwen/Qwen2.5-0.5B", description="HuggingFace model name or path")


class OptimizerConfig(BaseModel):
    """AdamW optimizer config"""

    type: Literal["adamw"] = "adamw"
    lr: float = Field(3e-4, gt=0, description="Learning rate")
    weight_decay: float = Field(0.1, ge=0, description="Weight decay")
    betas: tuple[float, float] = Field((0.9, 0.95), description="Adam betas")


class SchedulerConfig(BaseModel):
    """LR scheduler config"""

    type: Literal["cosine"] = "cosine"
    warmup_steps: int = Field(100, ge=0, description="Warmup steps")
    min_lr_ratio: float = Field(
        0.1, ge=0, le=1, description="Minimum LR as ratio of max LR"
    )


class DataConfig(BaseModel):
    """Base config for data loading"""

    batch_size: int = Field(32, ge=1, description="Training batch size")
    micro_batch_size: int = Field(
        8, ge=1, description="Micro batch size for gradient accumulation"
    )
    seq_len: int = Field(2048, ge=1, description="Sequence length")
    num_workers: int = Field(0, ge=0, description="Number of data loading workers")

    @model_validator(mode="after")
    def validate_batch_size(self):
        if self.batch_size % self.micro_batch_size != 0:
            raise ValueError("batch_size must be divisible by micro_batch_size")
        return self


class PretrainDataConfig(DataConfig):
    """Pretraining data config"""

    dataset_name: str = Field("stas/c4-en-10k", description="HF dataset name")
    split: str = Field("train", description="Dataset split")


class SFTDataConfig(DataConfig):
    """SFT data config"""

    dataset_name: str = Field(
        "trl-internal-testing/dolly-chatml-sft", description="HuggingFace dataset name"
    )
    split: str = Field("train_sft", description="Dataset split")
    packing: bool = Field(True, description="Pack multiple conversations per sequence")


class CheckpointConfig(BaseModel):
    """Checkpoint saving config"""

    save_interval: int = Field(1000, gt=0, description="Steps between checkpoint saves")
    keep_latest: int = Field(
        3, gt=0, description="Number of recent checkpoints to keep"
    )
    save_dir: Path = Field(
        Path("checkpoints"), description="Directory to save checkpoints"
    )


class PretrainConfig(BaseSettings):
    """Pretraining config"""

    model: ModelConfig = ModelConfig()
    data: PretrainDataConfig = PretrainDataConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()

    max_steps: int = Field(10000, gt=0, description="Maximum training steps")
    log_interval: int = Field(10, gt=0, description="Steps between log outputs")
    eval_interval: int = Field(1000, gt=0, description="Steps between evaluations")
    gradient_accumulation_steps: int = Field(
        1, gt=0, description="Gradient accumulation steps"
    )
    gradient_clipping: float = Field(1.0, gt=0, description="Gradient clipping norm")

    model_config = SettingsConfigDict(
        cli_parse_args=True,
        cli_kebab_case=True,
        cli_implicit_flags=True,
    )


class SFTConfig(BaseSettings):
    """SFT config"""

    model: ModelConfig = ModelConfig()
    data: SFTDataConfig = SFTDataConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()

    epochs: int = Field(1, gt=0, description="Number of epochs")
    log_interval: int = Field(10, gt=0, description="Steps between log outputs")
    eval_interval: int = Field(100, gt=0, description="Steps between evaluations")
    gradient_accumulation_steps: int = Field(
        1, gt=0, description="Gradient accumulation steps"
    )
    gradient_clipping: float = Field(1.0, gt=0, description="Gradient clipping norm")

    model_config = SettingsConfigDict(
        cli_parse_args=True,
        cli_kebab_case=True,
        cli_implicit_flags=True,
    )
