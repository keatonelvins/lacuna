# ref: https://github.com/PrimeIntellect-ai/prime-rl/blob/main/src/prime_rl/trainer/custom_models/__init__.py
from collections import OrderedDict

from transformers import AutoConfig
from transformers.models.auto.auto_factory import _BaseAutoModelClass, _LazyAutoMapping, auto_class_update
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

from lacuna.models.qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM
from lacuna.models.qwen3 import Qwen3Config, Qwen3ForCausalLM

AutoConfig.register("qwen3_moe", Qwen3MoeConfig, exist_ok=True)
AutoConfig.register("qwen3", Qwen3Config, exist_ok=True)

_CUSTOM_CAUSAL_LM_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, OrderedDict())
_CUSTOM_CAUSAL_LM_MAPPING.register(Qwen3MoeConfig, Qwen3MoeForCausalLM, exist_ok=True)
_CUSTOM_CAUSAL_LM_MAPPING.register(Qwen3Config, Qwen3ForCausalLM, exist_ok=True)

class AutoLacunaModelForCausalLM(_BaseAutoModelClass):
    _model_mapping = _CUSTOM_CAUSAL_LM_MAPPING


AutoLacunaModelForCausalLM = auto_class_update(AutoLacunaModelForCausalLM, head_doc="shhh don't tell arthur")


__all__ = ["AutoLacunaModelForCausalLM"]
