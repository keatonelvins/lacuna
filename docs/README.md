# lacuna docs

Seeing smaller labs use `torchtitan` for pretraining and `axolotl` for post-training. These are awesome!

`lacuna` came from using these (and other frameworks) and not really understanding how they worked under the hood.

I want it to be tiny and hackable: `uv run count_lines` should return <3k

## Order of model builder
Liger/Kernelize -> AC -> torch.compile -> FSDP
- Model patches always happens first
- Compile wrapped AC: compile already recomputes with the min-cut partitioner, so wrapping AC over a compiled region might mean multiple recomputations
    - see https://pytorch.org/blog/activation-checkpointing-techniques/ for more info
- torch.compile before FSDP, otherwise FSDP2 wrapped modules would cause graph breaks

## Other considerations
- Don't apply weight decay to embeddings/layer norms/biases (https://github.com/karpathy/minGPT/pull/24#issuecomment-679316025)
- Only FA3 is supported for SFT w/ packing, SDPA needs some more work to get varlen attention working
- Prefer regional (layer-wise) over full model compilation (https://docs.pytorch.org/tutorials/recipes/regional_compilation.html)
- For FSDP, we fully shard the layers individually then finally the root model (https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html#how-to-use-fsdp2)
- We default to fp32 accumulation (`accum_dtype=torch.float32`) for stability reasons (at the cost of some speed/memory):
    - For Liger Kernel, can pass through starting in `0.6.2`: https://github.com/linkedin/Liger-Kernel/pull/830
- Set `OMP_NUM_THREADS` to `cpu_cores / num_gpus` (physical cores so no hyper-threads!!)

## Datasets
- We always use IterableDatasets. This is the default type from `load_dataset` if streaming, otherwise we call to `to_iterable_dataset()`
- If streaming, the number of dataset shards will match the number of remote parquets. Otherwise we manually set `num_shards` to your world size.
    - Having `num_shards == world_size` maximizes throughput using `split_dataset_by_node` as the dataset is split evenly across workers.
- Helpful docs
    - https://huggingface.co/docs/datasets/en/stream
    - https://huggingface.co/docs/datasets/en/use_with_pytorch
    - https://docs.pytorch.org/docs/stable/data.html

## Misc
- Some references that were helpful for me while writing this:
    - [Mesh Zoo](https://blog.ezyang.com/2025/08/the-parallelism-mesh-zoo/)
    - [ND-Parallel](https://huggingface.co/blog/accelerate-nd-parallel)
- Failed attempts:
    - Tried to use `FileSystemWriter(serialization_format=SerializationFormat.SAFETENSORS)` but ran into isseus with tied weights