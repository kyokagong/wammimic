#!/usr/bin/env bash
set -euo pipefail

source_run=$(realpath -- "${1:?Usage: bash scripts/launch_mimic_webrtc.sh SOURCE_RUN ACTIVE_RUN NAME INTERNAL_PORT [COMPARE_RUN]}")
active=$(realpath -- "${2:?Expected the checkpoint run}")
name=${3:?Expected the preview container name}
port=${4:?Expected an internal HTTP signaling port}
source=/workspace/blinkmorphmimic
preview_source=$(realpath -- "${MIMIC_PREVIEW_SOURCE:-$source_run/source}")
reference=/data/kk/blinkmorphmimic
image=sha256:991b7f453125a19e09db2ba64bba603beb096bf8c3e461931e687556eefa6989
volumes=(-v "$source_run:$source_run:rw" -v "$source_run/cache-preview:/home/isaaclab/.cache/warp:rw")
if [[ "$active" != "$source_run" ]]; then volumes+=(-v "$active:$active:ro"); fi
comparison=()
pico=()
if [[ -n "${MIMIC_PICO_DATA_ROOT:-}" ]]; then
  mkdir -p -- "$MIMIC_PICO_DATA_ROOT"
  pico_root=$(realpath -- "$MIMIC_PICO_DATA_ROOT")
  volumes+=(-v "$pico_root:$pico_root:rw")
  pico=(--pico-data-root "$pico_root")
fi
if [[ $# == 5 ]]; then
  compare=$(realpath -- "$5")
  comparison=(--compare-run "$compare")
  volumes+=(-v "$compare:$compare:ro")
fi

docker run -d --name "$name" --network container:blinkmorph-preview --restart unless-stopped \
  --gpus device=1 --shm-size=16g -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,video \
  -e "PYTHONPATH=$source/src:$source" -e PYTHONDONTWRITEBYTECODE=1 -e MUJOCO_GL=egl \
  -e OMP_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=1 \
  -v "$preview_source:$source:ro" -v "$source_run/tools:/entry:ro" "${volumes[@]}" \
  -v "$reference/prepared:$reference/prepared:ro" -v /data/kk/data:/data/kk/data:ro \
  -w "$source" --entrypoint python "$image" /entry/serve_mimic_preview.py \
  --run "$active" --host 0.0.0.0 --port "$port" --public-host 1.94.0.106 --nvenc "${comparison[@]}" "${pico[@]}"
