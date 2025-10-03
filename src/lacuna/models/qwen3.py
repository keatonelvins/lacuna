from typing import Optional, Union

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast

from fla.modules import RMSNorm
from fla.modules.mlp import GatedMLP
from liger_kernel.transformers.rope import liger_rotary_pos_emb
from fla.modules.fused_linear_cross_entropy import FusedLinearCrossEntropyLoss

from transformers.models.qwen3 import modeling_qwen3
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM


class Qwen3LacunaForCausalLM(Qwen3ForCausalLM):
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
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
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

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def lacuna_patch_qwen3() -> None:
    """Path Qwen3 ops with kernels from fla-core and liger kernel"""
    modeling_qwen3.apply_rotary_pos_emb = liger_rotary_pos_emb
    modeling_qwen3.Qwen3RMSNorm = RMSNorm
    modeling_qwen3.Qwen3ForCausalLM = Qwen3LacunaForCausalLM
    modeling_qwen3.Qwen3MLP = lambda config: GatedMLP(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        fuse_swiglu=True,
    )
