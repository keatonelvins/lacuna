# lacuna docs

## Order of model builder
Kernelize -> AC -> torch.compile -> FSDP
- Compile wrapped AC: compile already recomputes with the min-cut partitioner, so wrapping AC over a compiled region might mean multiple recomputations
    - see https://pytorch.org/blog/activation-checkpointing-techniques/ for more info
- torch.compile before FSDP, otherwise FSDP2 wrapped modules would cause graph breaks

## Other considerations
- Don't apply weight decay to embeddings/layer norms/biases (https://github.com/karpathy/minGPT/pull/24#issuecomment-679316025)
- Prefer regional (layer-wise) over full model compilation (https://docs.pytorch.org/tutorials/recipes/regional_compilation.html)
- For FSDP, we fully shard the layers individually, then finally the root model (https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html#how-to-use-fsdp2)
- Use fp32 for accum! Liger Kernel doesn't do this by default.
- AdamW params taken from https://arxiv.org/pdf/2509.02046 but you should tune!!
- When using the DistributedSampler, you must call .set_epoch() BEFORE casting the dataloader to an iterable, otherwise epoch reshuffles won't work.
- Follow https://arxiv.org/pdf/2404.10830 for best-fit packing and intra-document masking
    - This means each minibatch is converted to one long sample with no padding and masking support via varlen attention.
    - Also supported by https://arxiv.org/pdf/2503.15450!

## Checkpointing, Steps, and Epochs

Training can be configured with either `steps` OR `epochs` (not both):
```toml
[trainer]
steps = 100      # Train for exactly 100 steps (overrides epochs)
# OR
epochs = 3       # Train for 3 full passes through the dataset
```

### Checkpoint Resumption Behavior

| Config | Scheduler Loaded? | Dataloader Loaded? | Step Behavior |
|--------|------------------|-------------------|---------------|
| **Initial training with `steps`** | N/A | N/A | Steps: 1 → N |
| **Initial training with `epochs`** | N/A | N/A | Steps: 1 → (dataset.length × epochs) |
| **Resume + scheduler + dataloader** | ✅ Yes | ✅ Yes | Continues from checkpoint (e.g., 100 → 101) |
| **Resume + scheduler - dataloader** | ✅ Yes | ❌ No | Continues from checkpoint, fresh data |
| **Resume - scheduler + dataloader** | ❌ No | ✅ Yes | **RESETS to 0** → 1 → N (see note below) |
| **Resume - scheduler - dataloader** | ❌ No | ❌ No | **RESETS to 0** → 1 → N (annealing mode) |

**Important:** Excluding the scheduler **resets the step counter to 0**. This enables fresh training phases where the new scheduler needs to start from scratch for warmup/decay to work correctly. Use this for annealing or multi-stage training.

### Advanced: Excluding Components

You can exclude specific components from loading to enable advanced training scenarios:

#### Exclude Dataloader Only
Resume with new data while keeping optimizer momentum and LR schedule:
```toml
[checkpoint]
resume_from = "weights/step_100"
exclude_from_loading = ["dataloader"]
```

**Behavior:**
- Step counter continues (e.g., 100 → 101 → ...)
- Optimizer state preserved (momentum, variance)
- Scheduler continues from checkpoint position
- Fresh dataloader starts from beginning of new dataset

**Use cases:**
- Fine-tuning on different data
- Switching dataset splits
- Continuing training on new data sources

See `configs/test_checkpoint_finetune.toml` for an example.

#### Exclude Scheduler + Dataloader (Annealing)
Start a fresh training phase with new scheduler and data:
```toml
[checkpoint]
resume_from = "weights/step_100"
exclude_from_loading = ["scheduler", "dataloader"]

[trainer]
steps = 50  # New phase: 50 steps

[scheduler]
warmup_ratio = 0.2  # Fresh warmup schedule
```

**Behavior:**
- **Step counter RESETS to 0** (enables fresh training phase!)
- New scheduler starts from scratch (warmup, constant, decay)
- Fresh dataloader with new data
- Model weights and optimizer state preserved

**Use cases:**
- Annealing with fresh LR schedule and lower base LR
- Multi-stage training (pretrain → anneal → fine-tune)
- Resetting warmup for stability after checkpoint resume

**Important:** Excluding the scheduler triggers a step reset. This is intentional - the new scheduler needs to start from step 0 for warmup/decay to work correctly.

See `configs/test_checkpoint_anneal.toml` for an example.

## Datasets
- We will tokenize and pack using `.map()` from `datasets` which automatically fingerprints and caches the final dataset
    - This means you only need to tokenize and pack once! Subsequent runs will stream directly from the cache under `HF_HOME`.
    - Also, changing the seq_len will only re-pack instead of also re-tokenizing :)
    - If you change models but want to use the cache, you can pass in data.override_tokenizer to force cache use
- The DatasetConfig is passed directly through to `load_dataset`, so you can load datasets directly from s3 with s3fs:
    ```toml
    [[data.datasets]]
    path = "parquet"
    data_files = "s3://path/to/data/train/*.parquet"
    ```
- You can train on a subset of examples with slice notation, e.g. `split = "train[:1000]"`

## Tokenization
- HF tokenization is littered with footguns, highly recommend Quentin's post on this:
    - https://huggingface.co/blog/qgallouedec/gotchas-in-tokenizer-behavior
- To support assistant-only loss, we require the assistant completions be wrapped in `{% generation %}` tags (similar to TRL)
    - This lets us use the hf-native `return_assistant_tokens_mask` and avoid manually building.

## Misc
- Some references that were helpful for me while writing this:
    - [Mesh Zoo](https://blog.ezyang.com/2025/08/the-parallelism-mesh-zoo/)
    - [ND-Parallel](https://huggingface.co/blog/accelerate-nd-parallel)
    - [Visualizing 6D Mesh Parallelism](https://main-horse.github.io/posts/visualizing-6d/)
    - [llm.c](https://github.com/karpathy/llm.c)
    - [PyTorch Tuning Guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)