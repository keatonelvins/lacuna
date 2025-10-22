Very good boiled down ref: https://github.com/LambdaLabsML/distributed-training-guide.git

## Meta Device Pattern

```python
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config)  # No memory allocated
# ... apply parallelisms (TP/FSDP/HSDP) ...
model.to_empty(device="cuda")  # Allocates uninitialized memory
load_pretrained_weights(model, model_name)  # Load via DCP
initialize_buffers(model)  # Initialize non-persistent buffers
```

- `to_empty()` allocates uninitialized memory for all tensors. Most weights get loaded from checkpoint/pretrained, but non-persistent buffers (not saved in safetensors) remain garbage:
    - `rotary_emb.inv_freq` - Must reinitialize from config
    - `mlp.tokens_per_expert` - torchtitan MoE buffer (`persistent=False`), we zero out
    - `mlp.expert_bias` - torchtitan MoE buffer (`persistent=True`), also zero out. Since this one is persistent but not hf-native, we drop during save_checkpoint

## Tips

- Use `@record` header to propagate dist errors: https://docs.pytorch.org/docs/stable/elastic/errors.html
- Compile args: https://docs.pytorch.org/docs/stable/generated/torch.compile.html
- `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" uv run train ...` can save a few gb if you are close to OOMing
- Also potential speedups by setting `OMP_NUM_THREADS=...` to physical cores // num processes
- For HSDP (2D mesh), use `mesh["dp"]` for loss reduction (`mesh.ndim > 1` detects HSDP vs FSDP)
- Lots of tricks with compile: 
    - https://docs.google.com/document/d/1y5CRfMLdwEoF1nTk9q8qEu1mgMUuUtvhklPKJ2emLU8
    - https://chaimrand.medium.com/maximizing-ai-ml-model-performance-with-pytorch-compilation-7cdf840202e6
