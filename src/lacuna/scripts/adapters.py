"""
Adapt torchtitan moe layers to/from hf format. (default hf -> tt, --to-hf means tt -> hf)

NOTE: only qwen3_moe supported for now. deepseek arch (glm, longcat, etc.) uses shared experts and has a e_score_correction_bias
"""

import re
import shutil
import argparse
from pathlib import Path

import torch
from loguru import logger
from safetensors import safe_open
from huggingface_hub import snapshot_download, save_torch_state_dict


# ref: https://github.com/PrimeIntellect-ai/prime-rl/blob/main/scripts/convert_moe_to_hf.py
def get_max_layer_num(state_dict: dict[str, torch.Tensor]) -> int:
    return max(int(i.split(".")[2]) for i in state_dict.keys() if "model.layers." in i) + 1


def is_config_file(p: Path) -> bool:
    return p.is_file() and "safetensors" not in str(p) and ".distcp" not in str(p)


def convert_tt_moe_to_hf(state_dict: dict[str, torch.Tensor]):
    logger.info("Converting tt MoE layers to HF format")
    num_layers = get_max_layer_num(state_dict)
    for i in range(num_layers):
        if f"model.layers.{i}.mlp.router.gate.weight" not in state_dict:
            continue

        # torchtitan-specific training buffers (we initialize them fresh before training)
        if f"model.layers.{i}.mlp.tokens_per_expert" in state_dict:
            del state_dict[f"model.layers.{i}.mlp.tokens_per_expert"]
        if f"model.layers.{i}.mlp.expert_bias" in state_dict:
            del state_dict[f"model.layers.{i}.mlp.expert_bias"]

        state_dict[f"model.layers.{i}.mlp.gate.weight"] = state_dict[f"model.layers.{i}.mlp.router.gate.weight"]
        del state_dict[f"model.layers.{i}.mlp.router.gate.weight"]

        # ungroup experts (experts.w{1,2,3} → experts.{j}.{gate,down,up}_proj.weight)
        if f"model.layers.{i}.mlp.experts.w1" in state_dict:
            w1 = state_dict[f"model.layers.{i}.mlp.experts.w1"]
            w2 = state_dict[f"model.layers.{i}.mlp.experts.w2"]
            w3 = state_dict[f"model.layers.{i}.mlp.experts.w3"]

            for j in range(w1.shape[0]):
                state_dict[f"model.layers.{i}.mlp.experts.{j}.gate_proj.weight"] = w1[j].clone()
                state_dict[f"model.layers.{i}.mlp.experts.{j}.down_proj.weight"] = w2[j].clone()
                state_dict[f"model.layers.{i}.mlp.experts.{j}.up_proj.weight"] = w3[j].clone()

            del state_dict[f"model.layers.{i}.mlp.experts.w1"]
            del state_dict[f"model.layers.{i}.mlp.experts.w2"]
            del state_dict[f"model.layers.{i}.mlp.experts.w3"]


def convert_hf_moe_to_tt(state_dict: dict[str, torch.Tensor]):
    logger.info("Converting HF MoE layers to torchtitan format")
    num_layers = get_max_layer_num(state_dict)

    for i in range(num_layers):
        if not any(f"model.layers.{i}.mlp.experts." in key for key in state_dict.keys()):
            continue

        expert_nums = set()
        for key in state_dict.keys():
            match = re.match(rf"model\.layers\.{i}\.mlp\.experts\.(\d+)\.", key)
            if match:
                expert_nums.add(int(match.group(1)))
        if not expert_nums:
            continue

        # group experts (experts.{j}.{gate,down,up}_proj.weight → experts.w{1,2,3})
        num_experts = max(expert_nums) + 1
        w1_list, w2_list, w3_list = [], [], []
        for j in range(num_experts):
            gate_key = f"model.layers.{i}.mlp.experts.{j}.gate_proj.weight"
            if gate_key in state_dict:
                w1_list.append(state_dict[gate_key])
                w3_list.append(state_dict[f"model.layers.{i}.mlp.experts.{j}.up_proj.weight"])
                w2_list.append(state_dict[f"model.layers.{i}.mlp.experts.{j}.down_proj.weight"])
                del state_dict[gate_key]
                del state_dict[f"model.layers.{i}.mlp.experts.{j}.up_proj.weight"]
                del state_dict[f"model.layers.{i}.mlp.experts.{j}.down_proj.weight"]

        if len(w1_list) != num_experts:
            raise ValueError(f"Layer {i}: Expected {num_experts} experts but found {len(w1_list)}")

        state_dict[f"model.layers.{i}.mlp.experts.w1"] = torch.stack(w1_list, dim=0)
        state_dict[f"model.layers.{i}.mlp.experts.w2"] = torch.stack(w2_list, dim=0)
        state_dict[f"model.layers.{i}.mlp.experts.w3"] = torch.stack(w3_list, dim=0)

        gate_key = f"model.layers.{i}.mlp.gate.weight"
        if gate_key in state_dict:
            state_dict[f"model.layers.{i}.mlp.router.gate.weight"] = state_dict[gate_key]
            del state_dict[gate_key]


def save_sharded_model(state_dict: dict[str, torch.Tensor], input_path: str, output_path: str):
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving sharded model to {output_dir}")
    save_torch_state_dict(
        state_dict,
        save_directory=output_dir,
        filename_pattern="model{suffix}.safetensors",
        max_shard_size="5GB",
        safe_serialization=True,
    )

    utils_paths = [p for p in Path(input_path).glob("*") if is_config_file(p)]
    logger.info(f"Copying {len(utils_paths)} config files to {output_dir}")
    for path in utils_paths:
        shutil.copy(path, output_dir / path.name)


def load_state_dict(input_path: str) -> tuple[dict[str, torch.Tensor], str]:
    """Load state dict from path, downloading from HF if needed."""
    if not Path(input_path).exists():
        input_path = snapshot_download(repo_id=input_path, repo_type="model")

    logger.info(f"Loading state dict from {input_path}")
    state_dict = {}
    for path in Path(input_path).glob("*.safetensors"):
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)

    return state_dict, input_path


def convert(input_path: str, output_path: str | None, to_hf: bool):
    if output_path is None:
        input_parent = Path(input_path).parent
        suffix = "hf" if to_hf else "lacuna"
        output_path = str(input_parent / suffix)

    state_dict, input_path = load_state_dict(input_path)
    if to_hf:
        convert_tt_moe_to_hf(state_dict)
    else:
        convert_hf_moe_to_tt(state_dict)
    save_sharded_model(state_dict, input_path, output_path)


def main():
    parser = argparse.ArgumentParser(description="Handle torchtitan moe layer adaption")
    parser.add_argument("input_path", type=str, help="Input model path")
    parser.add_argument(
        "output_path",
        type=str,
        default=None,
        help="Output path. Defaults to '/lacuna' or '/hf' in save dir (depending on --to-hf).",
    )
    parser.add_argument(
        "--to-hf", action="store_true", help="Convert from lacuna (tt moe) to HF format, otherwise hf -> lacuna"
    )
    args = parser.parse_args()

    convert(args.input_path, args.output_path, args.to_hf)


if __name__ == "__main__":
    main()
