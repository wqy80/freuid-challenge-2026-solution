#!/usr/bin/env bash
set -euo pipefail

CONTAINER="${1:-tianquan_LLM}"
OUT="${2:-environment/runtime}"
mkdir -p "$OUT"

if ! docker inspect "$CONTAINER" >/dev/null 2>&1; then
  echo "container not found: $CONTAINER" >&2
  exit 2
fi

{
  echo "container=$CONTAINER"
  docker inspect --format='image={{.Config.Image}}' "$CONTAINER"
  docker inspect --format='image_id={{.Image}}' "$CONTAINER"
  docker inspect --format='created={{.Created}}' "$CONTAINER"
  docker inspect --format='working_dir={{.Config.WorkingDir}}' "$CONTAINER"
  docker inspect --format='network_mode={{.HostConfig.NetworkMode}}' "$CONTAINER"
  docker inspect --format='ipc_mode={{.HostConfig.IpcMode}}' "$CONTAINER"
  docker inspect --format='shm_size={{.HostConfig.ShmSize}}' "$CONTAINER"
  docker inspect --format='runtime={{.HostConfig.Runtime}}' "$CONTAINER"
  docker inspect --format='restart_policy={{.HostConfig.RestartPolicy.Name}}' "$CONTAINER"
  echo "mount destinations:"
  docker inspect --format='{{range .Mounts}}  {{.Type}} -> {{.Destination}}{{println}}{{end}}' "$CONTAINER"
} > "$OUT/container-summary.txt"

image="$(docker inspect --format='{{.Config.Image}}' "$CONTAINER")"
docker image inspect --format='{{json .RepoDigests}}' "$image" > "$OUT/image-repo-digests.json"
docker history --no-trunc "$image" > "$OUT/image-history.txt"
nvidia-smi > "$OUT/host-nvidia-smi.txt"

echo "host/container record written to $OUT"
