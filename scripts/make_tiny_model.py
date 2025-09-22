"""
Usage: uv run scripts/make_tiny_model.py Qwen/Qwen3-30B-A3B-Base keatone/Qwen3-MoE-Tiny --push
"""

import sys
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

input_model_name = sys.argv[1]
output_model_name = sys.argv[2]
push = "--push" in sys.argv

tokenizer = AutoTokenizer.from_pretrained(input_model_name)
config = AutoConfig.from_pretrained(input_model_name)

config.update(
    dict(
        head_dim=32,
        hidden_size=8,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
    )
)

tiny_model = AutoModelForCausalLM.from_config(config)
tiny_model.bfloat16()

if push:
    tiny_model.push_to_hub(output_model_name)
    config.push_to_hub(output_model_name)
    tokenizer.push_to_hub(output_model_name)
else:
    tiny_model.save_pretrained(output_model_name)
    config.save_pretrained(output_model_name)
    tokenizer.save_pretrained(output_model_name)
