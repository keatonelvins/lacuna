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