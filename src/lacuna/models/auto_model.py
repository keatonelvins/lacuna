from transformers import AutoConfig
from transformers import AutoModelForCausalLM

from lacuna.models.qwen3 import lacuna_patch_qwen3
from lacuna.models.qwen3_moe import lacuna_patch_qwen3_moe


class AutoLacunaModelForCausalLM(AutoModelForCausalLM):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        model_type = model_config.model_type

        if model_type == "qwen3":
            lacuna_patch_qwen3()
        elif model_type == "qwen3_moe":
            lacuna_patch_qwen3_moe()
        else:
            raise ValueError(f"Model type {model_type} not supported")

        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
