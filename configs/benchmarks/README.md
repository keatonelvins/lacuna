### Benchmark Stages
0. Naive HF Implementation
1. Use `kernelize()` to swap out RMSNorm and `torch.compile(mode=max_autotune)`
2. Swap out all core components using Liger Kernel
3. Try above but now with `torch.compile(mode=max_autotune)`
4. Reduce memory usage using `cut-cross-entropy` and Liger Kernel
5. Combine 3 & 4 for max tuning

### Notes
- All stages are run with pre-compiled Flash Attention 3 (from kernels-community/flash-attn3)
- `torch.compile` is used on each layer individually rather than the full model
    - this was due to compatibility issues, also used torchtitan as a reference
    - used `fullgraph=true` and `mode=max_autotune`, o.w. default args
- Ran on a single H100 SXM w/ CUDA 12.8

### Results

Config                         Status               MFU %      TFLOPS     Runtime(s)  
------------------------------------------------------------
00_baseline                    CUDA OOM             -          -          87.2        
00_fa3                         Success!             34.6       342.6      108.0       
00_sdpa                        Success!             34.8       343.7      93.9        
01_kernelize_fa3_compile       Success!             0.0        0.0        38.1        
01_kernelize_sdpa_compile      Success!             44.6       440.8      146.7       
02_liger_fa3                   Success!             25.1       248.2      112.6       
02_liger_sdpa                  Success!             25.0       247.6      139.4       
03_liger_fa3_compile           Success!             0.0        0.0        37.4        
03_liger_sdpa_compile          Success!             26.3       260.3      134.5       
04_liger_cce_fa3               Success!             40.3       398.7      108.0       
04_liger_cce_sdpa              CUDA OOM             -          -          34.9        
05_liger_cce_fa3_compile       Success!             0.0        0.0        36.5        
05_liger_cce_sdpa_compile      CUDA OOM             -          -          49.4        
============================================================
Results saved to: benchmark_results/benchmark_results_20250825_050033.json
Total runtime: 1261.8s
============================================================