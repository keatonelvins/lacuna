# lacuna docs

Seeing smaller labs use `torchtitan` for pretraining and `axolotl` for post-training. These are awesome!

`lacuna` came from using these (and other frameworks) and not really understanding how they worked under the hood.

## Order of model builder
Liger/Kernelize -> AC -> torch.compile -> FSDP
- Model patches always happens first
- Compile wrapped AC: compile already recomputes with the min-cut partitioner, so wrapping AC over a compiled region might mean multiple recomputations
    - see https://pytorch.org/blog/activation-checkpointing-techniques/ for more info
- torch.compile before FSDP, otherwise FSDP2 wrapped modules would cause graph breaks

## Other considerations
- Don't apply weight decay to embeddings/layer norms/biases (https://github.com/karpathy/minGPT/pull/24#issuecomment-679316025)
- Prefer regional (layer-wise) over full model compilation (https://docs.pytorch.org/tutorials/recipes/regional_compilation.html)
- For FSDP, we fully shard the layers individually, then finally the root model (https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html#how-to-use-fsdp2)
- We default to fp32 accumulation (`accum_dtype=torch.float32`) for stability reasons (at the cost of some speed/memory):
    - For Liger Kernel, can pass through starting in `0.6.2`: https://github.com/linkedin/Liger-Kernel/pull/830
- Set `OMP_NUM_THREADS` to `cpu_cores / num_gpus` for torchrun (physical cores so no hyper-threads!!)
- Follow https://arxiv.org/pdf/2404.10830 for best-fit packing and intra-document masking
    - This means each minibatch is converted to one long sample with no padding and masking support via varlen attention.
    - Also supported by https://arxiv.org/pdf/2503.15450!
    - On top of intra-document masking, we also mask the first token on the boundary (TODO: run ablation)

## Datasets
- We will tokenize and pack using `.map()` from `datasets` which automatically fingerprints and caches the final dataset
    - This means you only need to tokenize and pack once! Subsequent runs will stream directly from the cache under `HF_HOME`.
    - Also, changing the seq_len will only re-pack instead of also re-tokenizing :)
    - If you change models but want to use the cache, you can pass in data.tokenizer_override to force cache use
- The DatasetConfig is passed directly through to `load_dataset`, so you can load datasets directly from s3 with s3fs:
    ```toml
    [[data.datasets]]
    path = "parquet"
    data_files = "s3://path/to/data/train/*.parquet"
    ```

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