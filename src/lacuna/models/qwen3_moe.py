from typing import Optional, Union

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast

from fla.modules import RMSNorm
from fla.modules.mlp import GatedMLP
from liger_kernel.transformers.rope import liger_rotary_pos_emb
from fla.modules.fused_linear_cross_entropy import FusedLinearCrossEntropyLoss
from torchtitan.models.moe import MoE, MoEArgs

from transformers.models.qwen3_moe import modeling_qwen3_moe
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeForCausalLM, 
    load_balancing_loss_func, 
    Qwen3MoeDecoderLayer, 
    Qwen3MoeConfig
)


class Qwen3MoeLacunaDecoderLayer(Qwen3MoeDecoderLayer):
    def __init__(self, config: Qwen3MoeConfig, layer_idx: int):
        super().__init__()

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
        else:
            self.mlp = GatedMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                fuse_swiglu=True,
            )


class Qwen3MoeLacunaForCausalLM(Qwen3MoeForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.flce = FusedLinearCrossEntropyLoss()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeCausalLMOutputWithPast:

        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )

        outputs: MoeModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_router_logits=output_router_logits,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        loss, logits = None, None

        # with FLCE, we don't materialize the logits during training to save memory
        if not self.training:
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.lm_head(hidden_states[:, slice_indices, :])

        if labels is not None:
            labels = nn.functional.pad(labels, (0, 1), value=self.flce.ignore_index)
            shifted_labels = labels[..., 1:].contiguous().to(hidden_states.device)
            loss = self.flce(hidden_states, shifted_labels, self.lm_head.weight, None)

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )


def lacuna_patch_qwen3_moe() -> None:
    """Path Qwen3 ops with kernels from fla-core and liger kernel"""
    modeling_qwen3_moe.apply_rotary_pos_emb = liger_rotary_pos_emb
    modeling_qwen3_moe.Qwen3RMSNorm = RMSNorm
    modeling_qwen3_moe.Qwen3MoeDecoderLayer = Qwen3MoeLacunaDecoderLayer
    modeling_qwen3_moe.Qwen3MoeForCausalLM = Qwen3MoeLacunaForCausalLM
