"""
Convert local DCP checkpoints to HuggingFace format.

Usage: 
uv run scripts/dcp_to_hf.py --ckpt-dir weights/model_name/step_5 --output-dir model_name_hf
"""

import pickle
import argparse
import json
import shutil
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from loguru import logger
from huggingface_hub import save_torch_state_dict


def load_dcp_checkpoint(ckpt_dir: Path) -> dict[str, torch.Tensor]:
    """Load DCP checkpoint and extract model state dict."""
    logger.info(f"Loading metadata from {ckpt_dir}")
    with open(ckpt_dir / ".metadata", "rb") as f:
        metadata = pickle.load(f)

    logger.info("Allocating empty tensors from metadata")
    state_dict = {"trainer": {"model": {}}}
    for k in metadata.state_dict_metadata.keys():
        if not k.startswith("trainer.model"):
            continue
        key = k.replace("trainer.model.", "")
        state_dict["trainer"]["model"][key] = torch.empty(
            metadata.state_dict_metadata[k].size, dtype=torch.bfloat16
        )

    logger.info(f"Loading checkpoint from {ckpt_dir}")
    dcp.load(state_dict, checkpoint_id=str(ckpt_dir))

    return state_dict["trainer"]["model"]


def save_hf_model(state_dict: dict[str, torch.Tensor], ckpt_dir: Path, output_dir: Path):
    """Save model in HuggingFace format with config files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving sharded model to {output_dir}")
    save_torch_state_dict(
        state_dict,
        save_directory=output_dir,
        filename_pattern="model{suffix}.safetensors",
        max_shard_size="5GB",
        safe_serialization=True,
    )

    utils_paths = [p for p in ckpt_dir.glob("*") if "safetensors" not in str(p) and ".distcp" not in str(p) and p.is_file()]
    logger.info(f"Copying {len(utils_paths)} config files to {output_dir}")
    for path in utils_paths:
        shutil.copy(path, output_dir / path.name)


def main(ckpt_dir: str, output_dir: str):
    ckpt_path = Path(ckpt_dir)
    state_dict = load_dcp_checkpoint(ckpt_path)
    save_hf_model(state_dict, ckpt_path, Path(output_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", type=str, required=True, help="Directory containing DCP checkpoint files")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for HF model")
    args = parser.parse_args()

    main(args.ckpt_dir, args.output_dir)