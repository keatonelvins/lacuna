"""MoE patching with torchtitan grouped mm layer."""

from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeDecoderLayer, Qwen3MoeConfig
from transformers.models.qwen3_moe import modeling_qwen3_moe
from torchtitan.models.moe import MoE, MoEArgs


# ref: https://huggingface.co/Qwen/Qwen3-30B-A3B-Base/blob/main/config.json
def get_qwen3_moe_args(config: Qwen3MoeConfig, use_grouped_mm: bool = True) -> MoEArgs:
    return MoEArgs(
        num_experts=config.num_experts,
        num_shared_experts=0,
        top_k=config.num_experts_per_tok,
        score_func="softmax",
        route_norm=config.norm_topk_prob,
        route_scale=1.0,
        score_before_experts=False,
        use_grouped_mm=use_grouped_mm,
    )


# ref: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py
class Qwen3MoeLacunaDecoderLayer(Qwen3MoeDecoderLayer):
    def __init__(self, config: Qwen3MoeConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        moe_args = get_qwen3_moe_args(config)

        if (layer_idx not in config.mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = MoE(moe_args, dim=config.hidden_size, hidden_dim=config.moe_intermediate_size)


def apply_tt_moe(model_type: str) -> None:
    """Patch model MoE layer with torchtitan implementation."""
    if model_type == "qwen3_moe":
        modeling_qwen3_moe.Qwen3MoeDecoderLayer = Qwen3MoeLacunaDecoderLayer
