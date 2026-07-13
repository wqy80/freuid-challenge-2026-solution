#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-environment/runtime}"
mkdir -p "$OUT"

uname -a > "$OUT/uname.txt"
cat /etc/os-release > "$OUT/os-release.txt"
python3 --version > "$OUT/python-version.txt" 2>&1
python3 -m pip freeze --all | sort > "$OUT/pip-freeze.txt"
nvidia-smi > "$OUT/nvidia-smi.txt"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader \
  > "$OUT/gpus.csv"

python3 - <<'PY' > "$OUT/python-runtime.json"
import json
import platform

import numpy
import pandas
import PIL
import timm
import torch
import torchvision

print(json.dumps({
    "platform": platform.platform(),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "torch_cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
    "timm": timm.__version__,
    "numpy": numpy.__version__,
    "pandas": pandas.__version__,
    "pillow": PIL.__version__,
}, indent=2))
PY

echo "environment written to $OUT"
