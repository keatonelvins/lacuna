"""Fuse sharded DCP checkpoints into consolidated hf format."""

import argparse
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from loguru import logger
from transformers import AutoModelForCausalLM

from lacuna.scripts.adapters import save_sharded_model


# TODO: untested with meta device, may also need to patch tt moe layer
def load_dcp_checkpoint(ckpt_dir: Path) -> dict[str, torch.Tensor]:
    """Load DCP checkpoint and extract model state dict."""
    logger.info(f"Loading checkpoint from {ckpt_dir}")
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(ckpt_dir / "config.json")

    state_dict = {"model": model.state_dict()}
    dcp.load(state_dict, checkpoint_id=str(ckpt_dir))

    return state_dict["trainer"]["model"]


def convert_checkpoint(ckpt_dir: str, output_dir: str):
    ckpt_path = Path(ckpt_dir)
    state_dict = load_dcp_checkpoint(ckpt_path)
    save_sharded_model(state_dict, ckpt_path, Path(output_dir))


def main():
    parser = argparse.ArgumentParser(description="Convert a DCP checkpoint directory to a hf safetensors format")
    parser.add_argument("ckpt_dir", type=str, help="Directory containing DCP checkpoint files")
    parser.add_argument("output_dir", type=str, help="Output directory for HF model")
    args = parser.parse_args()

    convert_checkpoint(args.ckpt_dir, args.output_dir)


if __name__ == "__main__":
    main()
