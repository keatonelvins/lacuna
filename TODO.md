# lacuna Training Framework TODO

## Philosophy: Opinionated & Correct
- **No abstractions** without clear value
- **Best practices only** - AdamW, cosine schedule, mixed precision
- **HF models directly** - no registry needed
- **<100k tokens total** - every line counts

## Architecture (9 files, ~1067 lines total)

```
lacuna/
├── src/lacuna/
│   ├── __init__.py         # Package init - 3 lines
│   ├── config.py           # Pydantic configs (pretrain + sft) - 150 lines
│   ├── cli.py              # Entry point, arg parsing, torchrun - 82 lines
│   ├── trainer.py          # Core training loop - 173 lines
│   ├── data.py             # Dataloader, tokenization, packing - 472 lines
│   ├── checkpoint.py       # Save/load with torch.save - 69 lines
│   ├── metrics.py          # MFU, memory, throughput (TODO) - 11 lines
│   ├── distributed.py      # FSDP setup, world info - 86 lines
│   └── utils.py            # Logging, misc helpers - 21 lines
├── configs/
│   ├── pt_qwen.toml        # Integration test config (TinierStories, 20 steps)
│   └── sft_qwen.toml       # Integration test config (tiny dataset)  
└── TODO.md                 # This file
```

## Implementation Phases

### Phase 1: Minimal Working Trainer ✅ COMPLETED
**Goal**: Train a small model end-to-end on single GPU

- [x] **Config System** ✅
  - [x] Pydantic models for PretrainConfig and SFTConfig in `config.py`
  - [x] TOML loading with CLI overrides in `cli.py`
  - [x] Opinionated defaults (AdamW, cosine, bf16)

- [x] **Model Loading** ✅
  - [x] AutoLigerKernelForCausalLM with automatic kernel optimizations
  - [x] Flash attention via `attn_implementation="flash_attention_2"`
  - [x] bfloat16 mixed precision, TF32 enabled

- [x] **Data Pipeline** ✅
  - [x] Simple non-streaming HF datasets loading in `data.py`
  - [x] Text concatenation and fixed-length chunking
  - [x] Proper input/label alignment for causal LM

- [x] **Training Loop** ✅
  - [x] Forward, backward, optimizer.step() in `trainer.py`
  - [x] Loss, throughput, memory logging every N steps
  - [x] Checkpoint saving with automatic cleanup in `checkpoint.py`

### Phase 2: Multi-GPU & Efficiency (Days 4-6)
**Goal**: Scale to 8 GPUs with good efficiency

- [ ] **FSDP Integration**
  - [ ] Auto-wrap based on transformer layers in `distributed.py`
  - [ ] Mixed precision with bf16
  - [ ] Gradient checkpointing for memory

- [X] **SFT with Packing** ✅ COMPLETED
  - [X] Parse OpenAI chat format in `data.py`
  - [X] Pack multiple conversations per batch (Best Fit Decreasing)
  - [X] Assistant-only loss masking with -100 labels

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

- [x] **CLI Entrypoints** ✅ COMPLETED
  - [x] `uv run pt` - Pretraining entry point (renamed from pretrain)
  - [x] `uv run sft` - Supervised fine-tuning entry point  
  - [x] `--torchrun` flag for distributed training
  - [ ] `uv run chat` - Terminal chat interface for testing models
  - [x] Support config file syntax like `uv run sft configs/sft_qwen.toml`
  - [x] Support CLI overrides like `--model.name`, `--torchrun.nproc-per-node`

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
- **FSDP** - Simpler than DDP+ZeRO, built into PyTorch

## Code Style
- Validate configs with Pydantic
- Clear variable names, no abbreviations
- Docstrings for public functions only
- Minimal comments in code (self-documenting)

## Integration Test Validation ✅ WORKING
```bash
# Integration test (validates setup, not for real experiments):
uv run pt configs/pt_qwen.toml

# For real pretraining experiments, use larger datasets:
uv run pt configs/pt_qwen.toml \
  --data.dataset-name roneneldan/TinyStories \
  --trainer.steps 10000 \
  --trainer.batch-size 32

# Results: ✅ Loss decreasing (10.97 → 2.84)
# Results: ✅ ~12k tokens/sec throughput  
# Results: ✅ 10.6GB memory on RTX 3090
# Results: ✅ Liger kernels auto-applied
# Results: ✅ Checkpoints saving/loading
```

## Success Metrics
- ✅ Full framework in ~1067 lines (target <2000)
- ✅ Configs in 150 lines (target <200)  
- ✅ Clean CLI with torchrun integration
- ✅ SFT with assistant masking and packing
- [ ] >50% MFU on H100 (need metrics.py implementation)
- [ ] Seamless multi-node scaling (FSDP TODO)
- ✅ Zero external dependencies beyond core libs

This approach prioritizes **getting the fundamentals perfect** over flexibility. We make the right choices so users don't have to think about them.