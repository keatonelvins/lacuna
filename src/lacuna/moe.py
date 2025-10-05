from transformers import AutoConfig
from transformers.models.qwen3_moe import modeling_qwen3_moe
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeDecoderLayer, 
    Qwen3MoeConfig
)

from liger_kernel.transformers.auto_model import AutoLigerKernelForCausalLM
from torchtitan.models.moe import MoE, MoEArgs


class Qwen3MoeLacunaDecoderLayer(Qwen3MoeDecoderLayer):
    def __init__(self, config: Qwen3MoeConfig, layer_idx: int):
        super().__init__(config, layer_idx)

        moe_args = MoEArgs(
            num_experts=config.num_experts,
            num_shared_experts=0,
            score_func="softmax",
            route_norm=config.norm_topk_prob,
            route_scale=1.0,
            score_before_experts=False,
            top_k=config.num_experts_per_tok,
            use_grouped_mm=True,
            load_balance_coeff=1e-3,
        )

        if (layer_idx not in config.mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = MoE(moe_args, dim=config.hidden_size, hidden_dim=config.moe_intermediate_size)


class AutoLacunaModelForCausalLM(AutoLigerKernelForCausalLM):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        model_type = model_config.model_type

        if model_type == "qwen3_moe":
            modeling_qwen3_moe.Qwen3MoeDecoderLayer = Qwen3MoeLacunaDecoderLayer
        else:
            raise ValueError(f"Model type {model_type} not supported")

        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)