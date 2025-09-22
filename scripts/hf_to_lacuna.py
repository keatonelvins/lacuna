"""
Usage: uv run scripts/hf_to_lacuna.py tiny-qwen
"""

import re
import sys
import torch
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

model_name = sys.argv[1]
output_model_name = f"{model_name}-Lacuna"

model = AutoModelForCausalLM.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


def patch_model(model):
    state_dict = model.state_dict()

    layers = [int(m.group(1)) for k in state_dict.keys() if (m := re.search(r"^model\.layers\.(\d+)\.", k)) is not None]

    for i in range(max(layers) + 1):
        has_gate = f"model.layers.{i}.mlp.gate.weight" in state_dict
        expert_keys = [
            k for k in state_dict.keys() if k.startswith(f"model.layers.{i}.mlp.experts.") and ".gate_proj.weight" in k
        ]
        if not (has_gate and expert_keys):
            continue

        state_dict[f"model.layers.{i}.mlp.router.gate.weight"] = state_dict[f"model.layers.{i}.mlp.gate.weight"]
        del state_dict[f"model.layers.{i}.mlp.gate.weight"]

        # TODO: check this one
        escb = f"model.layers.{i}.mlp.gate.e_score_correction_bias"
        if escb in state_dict:
            state_dict[f"model.layers.{i}.mlp.expert_bias"] = state_dict[escb]
            del state_dict[escb]

        se_g = f"model.layers.{i}.mlp.shared_experts.gate_proj.weight"
        se_d = f"model.layers.{i}.mlp.shared_experts.down_proj.weight"
        se_u = f"model.layers.{i}.mlp.shared_experts.up_proj.weight"
        if se_g in state_dict and se_d in state_dict and se_u in state_dict:
            state_dict[f"model.layers.{i}.mlp.shared_experts.w1.weight"] = state_dict[se_g]
            state_dict[f"model.layers.{i}.mlp.shared_experts.w2.weight"] = state_dict[se_d]
            state_dict[f"model.layers.{i}.mlp.shared_experts.w3.weight"] = state_dict[se_u]
            del state_dict[se_g]
            del state_dict[se_d]
            del state_dict[se_u]

        idxs = sorted({int(m.group(1)) for k in expert_keys if (m := re.search(r"experts\.(\d+)\.gate_proj\.weight$", k))})
        if not idxs:
            continue

        w1_list, w2_list, w3_list = [], [], []
        for j in idxs:
            base = f"model.layers.{i}.mlp.experts.{j}"
            w1_list.append(state_dict[f"{base}.gate_proj.weight"])  # (moe_dim, dim)
            w2_list.append(state_dict[f"{base}.down_proj.weight"])  # (dim, moe_dim)
            w3_list.append(state_dict[f"{base}.up_proj.weight"])  # (moe_dim, dim)

        state_dict[f"model.layers.{i}.mlp.experts.w1"] = torch.stack(w1_list, dim=0)
        state_dict[f"model.layers.{i}.mlp.experts.w2"] = torch.stack(w2_list, dim=0)
        state_dict[f"model.layers.{i}.mlp.experts.w3"] = torch.stack(w3_list, dim=0)

        for j in idxs:
            base = f"model.layers.{i}.mlp.experts.{j}"
            del state_dict[f"{base}.gate_proj.weight"]
            del state_dict[f"{base}.down_proj.weight"]
            del state_dict[f"{base}.up_proj.weight"]

    return state_dict


def patch_config(config):
    config.load_balance_coeff = 1e-3
    return config


def patch_tokenizer(tokenizer):
    return tokenizer


state_dict = patch_model(model)
config = patch_config(config)
tokenizer = patch_tokenizer(tokenizer)

model.save_pretrained(output_model_name, state_dict=state_dict)
config.save_pretrained(output_model_name)
tokenizer.save_pretrained(output_model_name)
