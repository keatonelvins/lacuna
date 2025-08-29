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

=========================================================================================
Config                         Status          MFU %    TFLOPS     Mem(GB)    Runtime(s)  
-----------------------------------------------------------------------------------------
00_baseline                    CUDA OOM        -        -          -          100.4       
00_fa2                         Success!        33.8     334.2      70.6       266.0       
00_fa3                         Success!        35.0     346.0      70.6       249.4       
00_sdpa                        Success!        34.8     343.9      71.2       240.2       
01_kernelize_fa2_compile       Success!        42.5     420.5      71.8       347.1       
01_kernelize_sdpa_compile      Success!        44.4     439.2      72.5       243.2       
02_liger_fa3                   Success!        25.1     248.2      69.4       296.6       
02_liger_sdpa                  Success!        25.1     247.8      70.0       295.3       
03_liger_fa2_compile           Success!        25.5     252.4      72.9       419.6       
03_liger_sdpa_compile          Success!        26.3     260.0      73.6       318.8       
04_liger_cce_fa3               Success!        40.5     400.4      67.0       216.5       
04_liger_cce_sdpa              Success!        40.1     396.1      67.6       216.9       
05_kernelize_cce_sdpa_compile  Success!        43.8     433.0      71.3       221.4       
05_liger_cce_fa2_compile       Success!        41.4     409.2      70.6       306.4       
05_liger_cce_sdpa_compile      Success!        43.3     428.3      71.3       223.0       
=========================================================================================
Results saved to: benchmark_results/benchmark_results_20250825_185902.json
Total runtime: 4208.5s
=========================================================================================

TODO: try torch_compile version of cce