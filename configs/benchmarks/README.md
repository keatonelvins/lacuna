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

========================================================================================
Config                         Status          MFU %    TFLOPS     Mem(GB)    Runtime(s)  
----------------------------------------------------------------------------------------
00_baseline                    CUDA OOM        -        -          -          91.7        
00_fa2                         Success!        33.8     333.9      70.6       105.3       
00_fa3                         Success!        34.8     344.5      70.6       120.8       
00_sdpa                        Success!        34.7     342.7      71.2       122.6       
01_kernelize_fa2_compile       Compile Issue   0.0      0.0        0.0        35.7        
01_kernelize_sdpa_compile      Success!        44.5     440.3      72.5       133.8       
02_liger_fa3                   Success!        25.1     248.7      69.4       133.3       
02_liger_sdpa                  Success!        25.1     247.8      70.0       136.4       
03_liger_fa2_compile           Compile Issue   0.0      0.0        0.0        35.2        
03_liger_sdpa_compile          Success!        26.3     260.5      73.6       152.7       
04_liger_cce_fa3               Success!        40.5     400.5      67.4       120.3       
04_liger_cce_sdpa              Success!        39.9     394.6      67.6       121.0       
05_liger_cce_fa2_compile       Compile Issue   0.0      0.0        0.0        45.0        
05_liger_cce_sdpa_compile      Success!        43.2     427.3      71.3       135.5       
========================================================================================
Results saved to: benchmark_results/benchmark_results_20250825_055428.json
Total runtime: 1495.8s
========================================================================================

* FA2 + compile should be possible in fullgraph=True but got data-dependent branching issue (transformers 4.55.4):
```bash
from user code:                                    
   File "/home/keaton_camfer_dev/lacuna/.venv/lib/python3.12/site-packages/transformers/modeling_layers.py", line 93, in __call__                                                                           
    return super().__call__(*args, **kwargs)                                                          
  File "/home/keaton_camfer_dev/lacuna/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl                                                                     
    return self._call_impl(*args, **kwargs)                                                           
  File "/home/keaton_camfer_dev/lacuna/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1784, in _call_impl                                                                             
    return forward_call(*args, **kwargs)           
  File "/home/keaton_camfer_dev/lacuna/.venv/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py", line 257, in forward                                                                
    hidden_states, _ = self.self_attn(             
  File "/home/keaton_camfer_dev/lacuna/.venv/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py", line 214, in forward                                                                
    attn_output, attn_weights = attention_interface(                                                  
  File "/home/keaton_camfer_dev/lacuna/.venv/lib/python3.12/site-packages/transformers/integrations/flash_attention.py", line 64, in flash_attention_forward                                                
    attn_output = _flash_attention_forward(                                                           
  File "/home/keaton_camfer_dev/lacuna/.venv/lib/python3.12/site-packages/transformers/modeling_flash_attention_utils.py", line 668, in _flash_attention_forward                                            
    elif is_fa_with_varlen_kwargs or is_fa_with_position_ids:   
```