import re
import torch
import pytest
from transformers import AutoConfig
from torchtitan.models.moe import MoE, MoEArgs
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
from transformers.models.glm4_moe.modeling_glm4_moe import Glm4MoeMoE

from lacuna.moe import get_qwen3_moe_args, get_glm4_moe_args
from lacuna.utils import set_seed


def convert_hf_moe_block(state_dict: dict[str, torch.Tensor], args: MoEArgs) -> dict[str, torch.Tensor]:
    """Convert a single hf moe block state_dict to tt moe format."""
    sd = state_dict.copy()

    # find expert indices
    expert_nums = set()
    for key in sd.keys():
        match = re.match(r"experts\.(\d+)\.", key)
        if match:
            expert_nums.add(int(match.group(1)))

    assert len(expert_nums) == args.num_experts, f"Expected {args.num_experts} experts but found {len(expert_nums)}"

    w1, w2, w3 = [], [], []
    for j in sorted(expert_nums):
        try:
            w1.append(sd.pop(f"experts.{j}.gate_proj.weight"))
            w3.append(sd.pop(f"experts.{j}.up_proj.weight"))
            w2.append(sd.pop(f"experts.{j}.down_proj.weight"))
        except KeyError as e:
            raise KeyError(f"Expert {j} incomplete: missing {e}")

    sd["experts.w1"] = torch.stack(w1, dim=0)
    sd["experts.w2"] = torch.stack(w2, dim=0)
    sd["experts.w3"] = torch.stack(w3, dim=0)

    if "gate.weight" in sd:
        sd["router.gate.weight"] = sd.pop("gate.weight")

    return sd


def check_block_equivalence(hf_moe: torch.nn.Module, tt_args: MoEArgs, config: AutoConfig):
    torch.backends.cuda.matmul.allow_tf32 = False
    set_seed(0)

    device = torch.device("cuda")
    dtype = torch.float32

    hf_moe = hf_moe.to(device=device, dtype=dtype)
    hf_state = {k: v.to(device=device, dtype=dtype) for k, v in hf_moe.state_dict().items()}

    tt_moe = MoE(tt_args, dim=config.hidden_size, hidden_dim=config.moe_intermediate_size)
    tt_moe = tt_moe.to(device=device, dtype=dtype)
    
    tt_state = convert_hf_moe_block(hf_state, tt_args)
    tt_moe.load_state_dict(tt_state, strict=False)

    x = torch.randn(2, 8, config.hidden_size, device=device, dtype=dtype)
    with torch.no_grad():
        y_hf = hf_moe(x)[0].to(dtype)
        y_tt = tt_moe(x).to(dtype)

    assert torch.allclose(y_hf, y_tt, rtol=1e-5, atol=1e-8), "Outputs diverged"


@pytest.mark.gpu
def test_qwen3_moe_equivalence():
    if not torch.cuda.is_available():
        pytest.skip("GPU not available")

    config = AutoConfig.from_pretrained("Qwen/Qwen3-30B-A3B-Base")
    hf_moe = Qwen3MoeSparseMoeBlock(config)
    tt_args = get_qwen3_moe_args(config, use_grouped_mm=False)  # grouped mm happens in bf16
    check_block_equivalence(hf_moe, tt_args, config)


@pytest.mark.gpu
@pytest.mark.xfail(reason="Adapter needs shared experts + score correction bias")
def test_glm4_moe_equivalence():
    if not torch.cuda.is_available():
        pytest.skip("GPU not available")

    config = AutoConfig.from_pretrained("zai-org/GLM-4.5-Air")
    hf_moe = Glm4MoeMoE(config)
    tt_args = get_glm4_moe_args(config, use_grouped_mm=False)  # grouped mm happens in bf16
    check_block_equivalence(hf_moe, tt_args, config)