#!/usr/bin/env bash
set -euo pipefail

run=$(realpath -- "${1:?Usage: bash scripts/finish_mimic_flips.sh RUN_DIR PARENT_RUN}")
parent=$(realpath -- "${2:?Expected the 10,000-step parent run}")
train=wammimic-mimic-flip-12000
source=/workspace/blinkmorphmimic
reference=/data/kk/blinkmorphmimic

exit_code=$(docker wait "$train")
if [[ "$exit_code" != 0 ]]; then
  echo "Training exited with status $exit_code" >&2
  exit 1
fi
image=$(docker inspect --format '{{.Image}}' "$train")
containers=()
for allocation in q1:0 bumi:2 g1:3; do
  robot=${allocation%:*}
  gpu=${allocation#*:}
  if [[ -s "$run/evaluation/flips/$robot-12000.json" ]]; then continue; fi
  container="wammimic-mimic-flip-eval-$robot"
  if docker inspect "$container" >/dev/null 2>&1; then docker rm "$container"; fi
  docker run -d --name "$container" --network none --gpus "device=$gpu" --shm-size=16g \
    -e "PYTHONPATH=$source/src:$source" -e PYTHONDONTWRITEBYTECODE=1 -e MUJOCO_GL=egl \
    -e OMP_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=1 \
    -v "$run/source:$source:ro" -v "$run/tools:/entry:ro" -v "$run:$run:rw" \
    -v "$parent/cache-warp:/home/isaaclab/.cache/warp:rw" \
    -v "$reference/prepared:$reference/prepared:ro" -v /data/kk/data:/data/kk/data:ro \
    -w "$source" --entrypoint python "$image" /entry/evaluate_mimic_flips.py \
    --run "$run" --output "$run/evaluation/flips" --robot "$robot"
  containers+=("$container")
done
status=0
for container in "${containers[@]}"; do
  if [[ $(docker wait "$container") != 0 ]]; then
    docker logs --tail 35 "$container" >&2
    status=1
  fi
done
exit "$status"
