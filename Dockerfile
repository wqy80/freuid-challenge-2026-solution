FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPYCACHEPREFIX=/submissions/.pycache \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HOME=/submissions/.hf \
    TORCH_HOME=/submissions/.torch \
    XDG_CACHE_HOME=/submissions/.cache \
    TMPDIR=/submissions/.tmp

WORKDIR /app

COPY requirements-inference.txt /app/requirements-inference.txt
RUN python -m pip install --no-cache-dir -r /app/requirements-inference.txt

COPY freuid /app/freuid
COPY scripts /app/scripts
COPY configs /app/configs
COPY docker /app/docker
COPY weights /app/weights

RUN find /app -type f -name '*.sh' -exec sed -i 's/\r$//' {} + \
    && chmod +x /app/docker/entrypoint.sh /app/docker/entrypoint_multigpu.sh \
    && python /app/scripts/check_release_weights.py --weights-dir /app/weights

ENTRYPOINT ["/app/docker/entrypoint.sh"]
