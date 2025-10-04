"""Misc utils for trainer and data."""

import time
import json
import torch
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from datetime import datetime
from pathlib import Path
from loguru import logger
from rich.pretty import Pretty
from rich.console import Console
from collections import defaultdict, deque
from torchtitan.components.metrics import build_device_memory_monitor
from torchtitan.tools.utils import get_peak_flops

from lacuna.distributed import is_master
from lacuna.config import LacunaConfig


def setup_env(config: LacunaConfig) -> Path:
    """Setup environment and output artifacts for a run."""
    # high -> TF32, highest -> FP32
    torch.set_float32_matmul_precision("high")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = setup_run_dir(config, timestamp)
    setup_logger(run_dir)
    display_config(config)
    save_settings(run_dir, config)
    config.checkpoint.prepare_save_dir(timestamp)  # clear save_dir if not resuming

    return run_dir


def setup_logger(run_dir: Path = None) -> None:
    """Setup logging to console and run directory."""
    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}",
        level="INFO",
        filter=lambda r: is_master(),
    )

    if run_dir:
        logger.add(
            run_dir / "run.log",
            format="{time:HH:mm:ss} | {level} | {message}",
            level="INFO",
            filter=lambda r: is_master(),
            rotation="100 MB",
        )


def cleanup_env() -> None:
    """Cleanup active run link."""
    active_link = Path(".lacuna_cache/active_run")
    if active_link.exists() or active_link.is_symlink():
        active_link.unlink()


def master_only(fn):
    """Decorator to run a function only on the master process."""

    def wrapper(*args, **kwargs):
        if not is_master():
            return
        return fn(*args, **kwargs)

    return wrapper


@master_only
def setup_run_dir(config: LacunaConfig, timestamp: str) -> Path:
    """Create and return a timestamped run directory."""
    run_dir = Path(".lacuna_cache/runs") / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    active_link = Path(".lacuna_cache/active_run")
    if active_link.exists():
        active_link.unlink()
    active_link.symlink_to(run_dir.relative_to(active_link.parent))

    return run_dir


@master_only
def display_config(config: LacunaConfig) -> None:
    """Pretty print the config in the console."""
    console = Console(force_terminal=False, no_color=True)
    with console.capture() as capture:
        console.print(Pretty(config, expand_all=True))  # omg Will you've outdone yourself
    logger.info("Starting training with config:\n" + capture.get().strip())


@master_only
def append_jsonl(run_dir: Path, metrics: dict, name: str = "metrics") -> None:
    metrics_data = {"timestamp": datetime.now().isoformat(), **metrics}
    metrics_file = run_dir / f"{name}.jsonl"
    with metrics_file.open("a") as f:
        f.write(json.dumps(metrics_data) + "\n")


@master_only
def log_training_metrics(step: int, metrics: dict, run_dir: Path) -> None:
    log_parts = [
        f"Step {step:>6}",
        f"Loss: {metrics['train/loss']:7.4f}",
        f"Grad: {metrics['train/grad_norm']:8.4f}",
        f"LR: {metrics['train/lr']:9.2e}",
        f"MFU: {metrics['perf/mfu(%)']:6.2f}%",
        f"Mem: {metrics['memory/max_active(GiB)']:6.2f}GiB",
        f"Toks: {metrics['train/ntokens_micro_batch']:4d}",
    ]
    logger.info(" | ".join(log_parts))
    append_jsonl(run_dir, metrics, "metrics")


@master_only
def log_eval_metrics(step: int, metrics: dict, run_dir: Path) -> None:
    log_parts = [
        f"Step {step:>6}",
        f"Eval Loss: {metrics['eval/loss']:7.4f}",
        f"Perplexity: {metrics['eval/perplexity']:9.3f}",
        f"Token Acc: {metrics['eval/token_accuracy'] * 100:6.2f}%",
    ]
    logger.info(" | ".join(log_parts))
    append_jsonl(run_dir, {"step": step, **metrics}, "eval")


@master_only
def save_settings(path: Path, config: LacunaConfig) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with (path / "settings.json").open("w") as f:
        f.write(config.model_dump_json(indent=4))


def log_loss_spikes(step: int, loss: float, model_inputs: dict, run_dir: Path) -> None:
    spiky_inputs = model_inputs["input_ids"].detach().cpu().tolist()
    append_jsonl(run_dir, {"step": step, "loss": loss, "model_inputs": spiky_inputs}, name="loss_spikes")


# ref: https://github.com/PrimeIntellect-ai/prime-rl/blob/main/src/prime_rl/trainer/perf.py
def get_active_mm_params(config) -> float:
    """Get number of active parameters per token involved in matmuls"""
    vocab_size = config.vocab_size
    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size
    head_dim = config.head_dim
    num_attention_heads = config.num_attention_heads
    num_hidden_layers = config.num_hidden_layers

    ## Attention
    if hasattr(config, "q_lora_rank") and hasattr(config, "kv_lora_rank"):
        # MLA
        q_params = num_hidden_layers * (
            hidden_size * config.q_lora_rank + config.q_lora_rank * num_attention_heads * config.qk_head_dim
        )
        kv_params = num_hidden_layers * (
            hidden_size * (config.kv_lora_rank + config.qk_rope_head_dim)
            + config.kv_lora_rank * num_attention_heads * (config.qk_nope_head_dim + config.v_head_dim)
        )
        o_params = num_hidden_layers * (num_attention_heads * config.v_head_dim * hidden_size)
    else:
        # GQA
        num_key_value_heads = config.num_key_value_heads
        q_params = num_hidden_layers * hidden_size * num_attention_heads * head_dim
        kv_params = 2 * num_hidden_layers * hidden_size * num_key_value_heads * head_dim
        o_params = num_hidden_layers * hidden_size * num_attention_heads * head_dim

    ## MLP
    if hasattr(config, "first_k_dense_replace"):
        num_dense_layers = config.first_k_dense_replace
        num_sparse_layers = config.num_hidden_layers - num_dense_layers
    elif hasattr(config, "num_experts_per_tok"):
        num_dense_layers = 0
        num_sparse_layers = config.num_hidden_layers
    else:
        num_dense_layers = config.num_hidden_layers
        num_sparse_layers = 0

    dense_mlp_params = num_dense_layers * 3 * intermediate_size * hidden_size
    sparse_mlp_params = 0
    if hasattr(config, "num_shared_experts"):  # Shared experts
        sparse_mlp_params += (
            num_sparse_layers * config.num_shared_experts * 3 * config.moe_intermediate_size * hidden_size
        )
    if hasattr(config, "num_experts_per_tok"):  # Routed experts
        sparse_mlp_params += (
            num_sparse_layers * config.num_experts_per_tok * 3 * config.moe_intermediate_size * hidden_size
        )
    if hasattr(config, "n_routed_experts"):  # DeepSeek Router
        sparse_mlp_params += num_sparse_layers * config.n_routed_experts * hidden_size
    elif hasattr(config, "num_experts"):  # Qwen Router
        sparse_mlp_params += num_sparse_layers * config.num_experts * hidden_size
    else:
        sparse_mlp_params = 0

    ## LM Head
    lm_head_params = vocab_size * hidden_size
    ## Total
    return q_params + kv_params + o_params + dense_mlp_params + sparse_mlp_params + lm_head_params


def calculate_model_flops(model: torch.nn.Module, seq_len: int) -> int:
    """Get parameter count and FLOPs/token at seq_len."""
    config = model.config
    l, h, q, t = (
        config.num_hidden_layers,
        config.num_attention_heads,
        config.hidden_size // config.num_attention_heads,
        seq_len,
    )
    # Reasoning behind the factor of 12 for the self-attention part of the formula:
    # 1. each self-attention has 2 matmul in the forward and 4 in the backward (6)
    # 2. the flash attention does 1 more matmul recomputation in the backward
    #    but recomputation should not be counted in calculating MFU           (+0)
    # 3. each matmul performs 1 multiplication and 1 addition                 (*2)
    # 4. we follow the convention and do not account for sparsity in causal attention
    flop_per_token = 6 * get_active_mm_params(config) + 12 * l * h * q * t

    return int(flop_per_token)

# ref: https://github.com/pytorch/torchtitan/blob/main/torchtitan/components/metrics.py
class MetricsProcessor:
    def __init__(self, config: LacunaConfig):
        self.config = config
        self.device_memory_monitor = build_device_memory_monitor()
        self.gpu_peak_flops = get_peak_flops(self.device_memory_monitor.device_name)
        self.ntokens_since_last_log = 0
        self.data_loading_times = []
        self.time_last_log = time.perf_counter()
        self.device_memory_monitor.reset_peak_stats()

    def get_metrics(self) -> dict:
        torch.cuda.synchronize()
        time_delta = time.perf_counter() - self.time_last_log
        tps = self.ntokens_since_last_log / time_delta
        mfu = 100 * self.num_flops_per_token * tps / self.gpu_peak_flops
        tflops = self.num_flops_per_token * tps / 1e12
        time_end_to_end = time_delta / self.config.metrics.log_every
        time_data_loading = sum(self.data_loading_times) / len(self.data_loading_times)
        time_data_loading_pct = 100 * sum(self.data_loading_times) / time_delta
        device_mem_stats = self.device_memory_monitor.get_peak_stats()

        metrics = {
            "perf/throughput(tps)": tps,
            "perf/tflops": tflops,
            "perf/mfu(%)": mfu,
            "time_metrics/end_to_end(s)": time_end_to_end,
            "time_metrics/data_loading(s)": time_data_loading,
            "time_metrics/data_loading_pct(%)": time_data_loading_pct,
            "memory/max_active(GiB)": device_mem_stats.max_active_gib,
            "memory/max_active_pct(%)": device_mem_stats.max_active_pct,
            "memory/max_reserved(GiB)": device_mem_stats.max_reserved_gib,
            "memory/max_reserved_pct(%)": device_mem_stats.max_reserved_pct,
            "memory/num_alloc_retries": device_mem_stats.num_alloc_retries,
            "memory/num_ooms": device_mem_stats.num_ooms,
        }

        self.ntokens_since_last_log = 0
        self.data_loading_times.clear()
        self.time_last_log = time.perf_counter()
        self.device_memory_monitor.reset_peak_stats()

        return metrics


def setup_metrics_processor(config: LacunaConfig, model: torch.nn.Module) -> MetricsProcessor:
    processor = MetricsProcessor(config)
    processor.num_flops_per_token = calculate_model_flops(model, config.trainer.seq_len)
    return processor


# some gpt-5 code for bfd packing
class IntSucc:
    __slots__ = ("N", "bits")

    def __init__(self, maxval: int):
        assert maxval >= 1
        self.N, self.bits = maxval, 0

    def add(self, i: int):
        self.bits |= 1 << (i - 1)

    def discard(self, i: int):
        self.bits &= ~(1 << (i - 1))

    def next_geq(self, x: int) -> int:
        y = self.bits >> (x - 1)
        assert y, "no successor present (missing sentinel?)"
        return x + ((y & -y).bit_length() - 1)


def _take(arr, idx):
    idx = np.asarray(idx, dtype=np.int32)
    out = pc.take(arr, pa.array(idx, type=pa.int32()))
    return out.combine_chunks() if isinstance(out, pa.ChunkedArray) else out


def pack_bfd(examples: pa.Table, seq_len: int, context_len: int | None = None, truncate: bool = True) -> pa.Table:
    """Drop or truncate examples longer than context_len then pack into long samples up to seq_len."""
    has_masks = "assistant_masks" in examples.column_names
    context_len = context_len or seq_len  # default to seq_len if context_len is not provided

    if truncate:
        ids = pc.list_slice(examples["input_ids"], 0, context_len)
        masks = pc.list_slice(examples["assistant_masks"], 0, context_len) if has_masks else None
    else: # drop samples longer than context_len
        sample_lens = pc.list_value_length(examples["input_ids"])
        long_sample_mask = pc.less_equal(sample_lens, context_len)
        num_kept, num_total = pc.sum(long_sample_mask).as_py(), len(sample_lens)
        logger.info(f"Rank {get_rank()}: dropped {num_total - num_kept} examples longer than context_len={context_len}")
        ids = pc.filter(examples["input_ids"], long_sample_mask)
        masks = pc.filter(examples["assistant_masks"], long_sample_mask) if has_masks else None

    lens = pc.list_value_length(ids).to_numpy()
    order = np.argsort(-lens)

    succ = IntSucc(seq_len)
    succ.add(seq_len)  # sentinel enables new bins
    by_space = defaultdict(deque)  # space -> deque[bins]
    bins = []  # each: {"ids": [...], "len": int}

    for i in order:
        L = int(lens[i])
        if not L:
            continue
        s = succ.next_geq(L)
        b = by_space[s].popleft() if s < seq_len else {"ids": [], "len": 0}
        if s < seq_len and not by_space[s]:
            succ.discard(s)
        b["ids"].append(int(i))
        b["len"] += L
        if s == seq_len:
            bins.append(b)
        ns = s - L
        by_space[ns].append(b)
        if ns:
            succ.add(ns)

    reorder = [j for b in bins for j in b["ids"]]
    ids_taken = _take(ids, reorder)
    if has_masks:
        masks_taken = _take(masks, reorder)

    # offsets (match ListArray vs LargeListArray via dtype)
    tok_counts = [b["len"] for b in bins]
    odtype = ids_taken.offsets.type.to_pandas_dtype()
    offs = np.cumsum([0] + tok_counts, dtype=odtype)

    LA = type(ids_taken)
    packed_ids = LA.from_arrays(offs, ids_taken.values)

    # position_ids: reset to 0 at each original example boundary
    dl = lens[reorder]
    T = int(offs[-1])
    pos = np.ones(T, dtype=np.int32)
    pos[0] = 0
    if dl.size > 1:
        cut = dl[:-1].cumsum()
        pos[cut] = -(dl[:-1] - 1)
    pos = pos.cumsum()
    position_ids = LA.from_arrays(offs, pa.array(pos, type=pa.int32()))

    if has_masks:
        packed_masks = LA.from_arrays(offs, masks_taken.values)
        return pa.Table.from_arrays(
            [packed_ids, position_ids, packed_masks], names=["input_ids", "position_ids", "assistant_masks"]
        )
    return pa.Table.from_arrays([packed_ids, position_ids], names=["input_ids", "position_ids"])
