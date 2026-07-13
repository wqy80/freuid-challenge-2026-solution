# Environment

The inference Docker image is defined by `Dockerfile` and the pinned packages in
`requirements-inference.txt`.

Training was run on four NVIDIA A40 GPUs. The exact server output must be
captured from the training container with:

```bash
bash scripts/capture_environment.sh environment/runtime
```

Run `scripts/capture_container_host.sh` on the Docker host to record the image,
shared-memory setting, network mode, GPU driver, and container destinations.
The script deliberately avoids dumping all environment variables.

The generated files are local records and are ignored by Git. Copy the useful
version numbers into this document after the capture is complete.

## Training runtime

- Operating system: pending capture
- Python: pending capture
- PyTorch: pending capture
- CUDA runtime: pending capture
- cuDNN: pending capture
- timm: 1.0.26
- GPUs: 4 x NVIDIA A40, 48 GB class

## Inference runtime

- Base image: `pytorch/pytorch:2.5.1-cuda11.8-cudnn9-runtime`
- Python dependencies: `requirements-inference.txt`
- Network access during inference: disabled
