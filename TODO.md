# lacuna Training Framework TODO

## Philosophy: Opinionated & Correct
- **No abstractions** without clear value
- **Best practices only** - AdamW, cosine schedule, mixed precision
- **HF models directly** - no registry needed
- **<100k tokens total** - every line counts

## Architecture (10 files, ~5k lines total)

```
lacuna/
├── src/lacuna/
│   ├── __init__.py         # Package init
│   ├── config.py           # Pydantic configs (pretrain + sft)
│   ├── train.py            # Entry point, arg parsing
│   ├── trainer.py          # Core training loop
│   ├── data.py             # Dataloader, tokenization, packing
│   ├── checkpoint.py       # Save/load with torch.save
│   ├── metrics.py          # MFU, memory, throughput
│   ├── distributed.py      # FSDP setup, world info
│   └── utils.py            # Logging, misc helpers
├── configs/
│   ├── pretrain_qwen.toml  # Pretrain config
│   └── sft_qwen.toml       # SFT config  
├── scripts/
│   └── launch.sh           # torchrun wrapper
└── TODO.md                 # This file
```

## Implementation Phases

### Phase 1: Minimal Working Trainer (Days 1-3)
**Goal**: Train a small model end-to-end on single GPU

- [ ] **Config System**
  - [ ] Pydantic models for PretrainConfig and SFTConfig in `config.py`
  - [ ] TOML loading with CLI overrides in `train.py`
  - [ ] Opinionated defaults (AdamW, cosine, bf16)

- [ ] **Model Loading**
  - [ ] Direct `AutoModelForCausalLM.from_pretrained()` in `trainer.py`
  - [ ] Automatic liger patching if available
  - [ ] Flash attention via `attn_implementation="flash_attention_2"`

- [ ] **Data Pipeline**
  - [ ] StreamingDataset from HF datasets in `data.py`
  - [ ] Simple tokenization with padding
  - [ ] Fixed sequence length for pretrain

- [ ] **Training Loop**
  - [ ] Forward, backward, optimizer.step() in `trainer.py`
  - [ ] Gradient accumulation
  - [ ] Loss logging every N steps
  - [ ] Simple checkpoint saving in `checkpoint.py`

### Phase 2: Multi-GPU & Efficiency (Days 4-6)
**Goal**: Scale to 8 GPUs with good efficiency

- [ ] **FSDP Integration**
  - [ ] Auto-wrap based on transformer layers in `distributed.py`
  - [ ] Mixed precision with bf16
  - [ ] Gradient checkpointing for memory

- [ ] **SFT with Packing**
  - [ ] Parse OpenAI chat format in `data.py`
  - [ ] Pack multiple conversations per batch (First Fit Decreasing)
  - [ ] Mask non-assistant tokens in loss

- [ ] **Performance Monitoring**
  - [ ] Calculate MFU based on model size in `metrics.py`
  - [ ] Track tokens/second throughput
  - [ ] Memory usage per GPU
  - [ ] WandB integration

### Phase 3: Production Ready (Days 7-9)
**Goal**: Robust training with monitoring

- [ ] **Checkpointing**
  - [ ] Save/resume with exact data position
  - [ ] Best checkpoint tracking
  - [ ] Automatic cleanup of old checkpoints

- [ ] **Multi-Node Support**
  - [ ] SLURM launcher script in `scripts/slurm.sh`
  - [ ] Handle node failures gracefully
  - [ ] Checkpoint to shared filesystem

- [ ] **CLI Entrypoints** (Following PRIME-RL pattern)
  - [ ] `uv run pretrain` - Pretraining entry point
  - [ ] `uv run sft` - Supervised fine-tuning entry point  
  - [ ] `uv run chat` - Terminal chat interface for testing models
  - [ ] Support `@` config file syntax like `uv run sft @ configs/sft_qwen.toml`
  - [ ] Support CLI overrides like `--model-name`, `--batch-size`

- [ ] **Terminal Chat Interface**
  - [ ] Simple terminal-based chat in `scripts/chat.py`
  - [ ] OpenAI-compatible endpoint support
  - [ ] Configurable temperature, top_p, max_tokens
  - [ ] Interactive terminal session with history

- [ ] **Testing & Docs**
  - [ ] Integration test with tiny model
  - [ ] Performance benchmarks
  - [ ] Usage examples in README

## Key Decisions (Opinionated!)

### Always Use:
- **AdamW** - Best optimizer for LLMs
- **Cosine schedule** with warmup - Works consistently
- **bf16 mixed precision** - Good range, fast on modern GPUs
- **Flash Attention 2** - When available (most GPUs now)
- **Liger kernels** - Auto-detect and apply
- **Gradient accumulation** - For effective batch size
- **FSDP** - Simpler than DDP+ZeRO, built into PyTorch

### Never Add (Keep It Simple):
- ❌ Multiple optimizers - AdamW is enough
- ❌ Complex schedulers - Cosine works great
- ❌ Model registry - HF names in config
- ❌ Factory patterns - Direct instantiation
- ❌ FP8 - Not worth complexity yet
- ❌ Pipeline/Tensor parallel - FSDP handles most cases
- ❌ Custom kernels - Liger/Flash are sufficient

## Code Style
- Type hints everywhere (jaxtyping for tensors)
- Validate configs with Pydantic
- Clear variable names, no abbreviations
- Docstrings for public functions only
- No comments in code (self-documenting)

## First PR Checklist
```bash
# Minimal trainer that can:
uv run pretrain @ configs/pretrain_qwen.toml \
  --model-name Qwen/Qwen2.5-0.5B \
  --batch-size 4 \
  --steps 100

# Should output:
# - Loss decreasing
# - Tokens/sec metric
# - Checkpoint saved
```

## Success Metrics
- ✅ Full pretrain in <1000 lines
- ✅ Full SFT in <500 lines  
- ✅ Configs in <200 lines
- ✅ >50% MFU on H100
- ✅ Seamless multi-node scaling
- ✅ Zero external dependencies beyond core libs

This approach prioritizes **getting the fundamentals perfect** over flexibility. We make the right choices so users don't have to think about them.