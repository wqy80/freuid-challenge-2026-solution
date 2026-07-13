# Environment

Training and final validation used four NVIDIA A40 GPUs. The final Docker test
ran on all 7,821 public images and produced valid output files.

## Validated runtime

- Operating system: Ubuntu 24.04.1 LTS
- Python: 3.12.3
- PyTorch: 2.10.0+cu126
- torchvision: 0.25.0+cu126
- CUDA used by PyTorch: 12.6
- cuDNN: 9.10.2
- NVIDIA driver: 550.90.07
- timm: 1.0.26
- NumPy: 2.4.4
- pandas: 3.0.3
- Pillow: 12.0.0
- GPUs: 4 x NVIDIA A40, 46,068 MiB each

The validated local image was `freuid2026:offline-code-freeze`, image ID
`sha256:10d3f01f566ed2baff6483d3c3269e6fb20db79a11850e8d5fcc4dd7c2516750`.
The full four-GPU test used 32 GiB shared memory and read-only input mounted at
`/data`. Output was written to `/submissions`.

The default single-GPU entry point passed a 32-image smoke test. The full
7,821-image check used the optional four-GPU entry point.

## Published Docker specification

The public `Dockerfile` uses
`pytorch/pytorch:2.5.1-cuda11.8-cudnn9-runtime` and installs the versions in
`requirements-inference.txt`. Model inference does not need network access once
the image and weights are available.

The public Dockerfile was also built from scratch on WSL2 Ubuntu 22.04. The
resulting image ID was
`sha256:69c2e013071ef72f40ae1c95446c6f3e02ccb345a131bac2c52c99d32f725f30`.
All eight weight files passed the size and SHA-256 checks during the build. The
default entry point then completed an end-to-end CPU smoke test on one real
image, including all eight checkpoints, candidate generation, and output
validation.

The local server had no internet access, so the full test used the existing
local runtime as the base image. The model code, frozen weights, entry points,
candidate generation, and output checks were the same as in this repository.

Raw environment records can be created with `scripts/capture_environment.sh`
and `scripts/capture_container_host.sh`. They are kept outside Git because the
full package list and Docker history are machine-specific.
