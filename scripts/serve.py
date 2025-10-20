# /// script
# dependencies = [
#     "sglang[all]>=0.5.3",
#     "flashinfer-cubin",
#     "flashinfer-python",
# ]
# ///

import os
import sys
import subprocess
from pathlib import Path


os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"  # set to 10.0 for blackwell

model_path = sys.argv[1]
model_name = Path(model_path).name

cmd = [
    "python -m sglang.launch_server",
    "--model-path",
    model_path,
    "--served-model-name",
    model_name,
    "--host",
    "0.0.0.0",
    "--port",
    str(8000),
    "--log-requests",
    "--log-requests-level",
    "2",
]

print("Launching: ", " ".join(cmd))
subprocess.run(" ".join(cmd), shell=True, check=True)
