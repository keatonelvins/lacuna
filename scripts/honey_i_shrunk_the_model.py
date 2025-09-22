"""
Usage: uv run scripts/honey_i_shrunk_the_model.py Qwen/Qwen3-30B-A3B-Base tiny-qwen
"""

import sys
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

input_model_name = sys.argv[1]
output_model_name = sys.argv[2]

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
tiny_model.save_pretrained(output_model_name)
config.save_pretrained(output_model_name)
tokenizer.save_pretrained(output_model_name)
