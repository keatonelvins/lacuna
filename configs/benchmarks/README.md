### Benchmark Stages
0. Naive HF Implementation
1. Use `kernelize()` to swap out RMSNorm and `torch.compile(mode=max_autotune)`
2. Swap out all core components using Liger Kernel
3. Try above but now with `torch.compile(mode=max_autotune)`
4. Reduce memory usage using `cut-cross-entropy` and Liger Kernel
5. Combine 3 & 4 for max tuning

All stages are run with Flash Attention 3 and SDPA.