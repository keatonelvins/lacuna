"""
Usage: uv run scripts/hf_to_lacuna.py keatone/Qwen3-MoE-Tiny --push
"""

import re
import sys
import torch
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

model_name = sys.argv[1]
output_model_name = f"{model_name}-Lacuna"
push = "--push" in sys.argv

model = AutoModelForCausalLM.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


def patch_model(model):
    state_dict = model.state_dict()

    # group the expert weights (gate_proj, up_proj, and down_proj) by layer/expert index
    grouped_experts = {}
    for sd_key, sd_value in state_dict.items():
        moe_match = re.match(
            r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$",
            sd_key,
        )
        if not moe_match:
            continue

        layer_idx, expert_idx, proj_type = int(moe_match.group(1)), int(moe_match.group(2)), moe_match.group(3)
        layer = grouped_experts.setdefault(layer_idx, {"w1": {}, "w2": {}, "w3": {}})
        if proj_type == "gate_proj":
            layer["w1"][expert_idx] = sd_value
        elif proj_type == "down_proj":
            layer["w2"][expert_idx] = sd_value
        else:  # up_proj
            layer["w3"][expert_idx] = sd_value

    for i, layer in grouped_experts.items():
        expert_idxs = sorted(layer["w1"].keys())

        # each layer has a router gate, add .router to key
        gate_key = f"model.layers.{i}.mlp.gate.weight"
        if gate_key in state_dict:
            state_dict[f"model.layers.{i}.mlp.router.gate.weight"] = state_dict.pop(gate_key)

        # inject grouped expert weights from above into the new state_dict
        state_dict[f"model.layers.{i}.mlp.experts.w1"] = torch.stack([layer["w1"][j] for j in expert_idxs], dim=0)
        state_dict[f"model.layers.{i}.mlp.experts.w2"] = torch.stack([layer["w2"][j] for j in expert_idxs], dim=0)
        state_dict[f"model.layers.{i}.mlp.experts.w3"] = torch.stack([layer["w3"][j] for j in expert_idxs], dim=0)

        for j in expert_idxs:
            del state_dict[f"model.layers.{i}.mlp.experts.{j}.gate_proj.weight"]
            del state_dict[f"model.layers.{i}.mlp.experts.{j}.down_proj.weight"]
            del state_dict[f"model.layers.{i}.mlp.experts.{j}.up_proj.weight"]

        # handle shared experts (torchtitan wants stacked tensors with shape [1, hidden_dim, dim])
        if f"model.layers.{i}.mlp.shared_experts.gate_proj.weight" in state_dict:
            state_dict[f"model.layers.{i}.mlp.shared_expert.w1"] = state_dict.pop(
                f"model.layers.{i}.mlp.shared_experts.gate_proj.weight"
            ).unsqueeze(0)
            state_dict[f"model.layers.{i}.mlp.shared_expert.w2"] = state_dict.pop(
                f"model.layers.{i}.mlp.shared_experts.down_proj.weight"
            ).unsqueeze(0)
            state_dict[f"model.layers.{i}.mlp.shared_expert.w3"] = state_dict.pop(
                f"model.layers.{i}.mlp.shared_experts.up_proj.weight"
            ).unsqueeze(0)

        # map e_score_correction_bias -> expert_bias
        expert_bias = f"model.layers.{i}.mlp.gate.e_score_correction_bias"
        if expert_bias in state_dict:
            state_dict[f"model.layers.{i}.mlp.expert_bias"] = state_dict.pop(expert_bias)
        else:
            state_dict[f"model.layers.{i}.mlp.expert_bias"] = torch.zeros(len(expert_idxs), dtype=torch.float32)

        # non-persistent buffer (used for load balancing during training)
        state_dict[f"model.layers.{i}.mlp.tokens_per_expert"] = torch.zeros(len(expert_idxs), dtype=torch.float32)

    return state_dict


def patch_config(config):
    config.load_balance_coeff = 1e-3
    return config


def patch_tokenizer(tokenizer):
    return tokenizer


state_dict = patch_model(model)
config = patch_config(config)
tokenizer = patch_tokenizer(tokenizer)

if push:
    model.push_to_hub(output_model_name, state_dict=state_dict)
    config.push_to_hub(output_model_name)
    tokenizer.push_to_hub(output_model_name)
else:
    model.save_pretrained(output_model_name, state_dict=state_dict)
    config.save_pretrained(output_model_name)
    tokenizer.save_pretrained(output_model_name)
