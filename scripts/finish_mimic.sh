#!/usr/bin/env bash
set -euo pipefail

run=$(realpath -- "${1:?Usage: bash scripts/finish_mimic.sh RUN_DIR [PREVIEW_HOST] [TRAIN_CONTAINER] [BASELINE_RUN]}")
preview_host=${2:-127.0.0.1}
train=${3:-wammimic-mimic-global-ref-10000}
baseline=${4:-}
source=/workspace/blinkmorphmimic
reference=/data/kk/blinkmorphmimic
scope=wammimic-mimic-global-ref
public=wammimic-mimic-preview

exit_code=$(docker wait "$train")
if [[ "$exit_code" != 0 ]]; then
  echo "Training container $train exited with status $exit_code" >&2
  exit 1
fi

train_image=$(docker inspect --format '{{.Image}}' "$train")
preview_image=sha256:991b7f453125a19e09db2ba64bba603beb096bf8c3e461931e687556eefa6989
runtime=(--shm-size=16g -e "PYTHONPATH=$source/src:$source" -e PYTHONDONTWRITEBYTECODE=1
  -e OMP_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=1 -e MUJOCO_GL=egl
  -v "$run/source:$source:ro" -v "$run:$run:rw" -v "$run/tools:/entry:ro"
  -v "$reference/prepared:$reference/prepared:ro" -v /data/kk/data:/data/kk/data:ro
  -w "$source" --entrypoint python)
warp=(-v "$run/cache-warp:/home/isaaclab/.cache/warp:rw")
preview_env=(--network host --gpus device=1 -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
  -v "$run/cache-preview:/home/isaaclab/.cache/warp:rw")

evaluate() {
  local stage=$1 target=$2
  shift 2
  local allocations=(q1:0 bumi:2 g1:3)
  local names=() allocation robot gpu name status=0
  local data_mount=()
  if [[ "$target" != "$run" ]]; then data_mount=(-v "$target:$target:ro"); fi
  for allocation in "${allocations[@]}"; do
    robot=${allocation%:*}
    gpu=${allocation#*:}
    name=$scope-$stage-$robot
    names+=("$name")
    if [[ $(docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null || true) != true ]]; then
      if docker inspect "$name" >/dev/null 2>&1; then docker rm "$name"; fi
      docker run -d --name "$name" --network none --gpus "device=$gpu" \
        "${data_mount[@]}" "${warp[@]}" "${runtime[@]}" "$train_image" \
        /entry/evaluate_mimic.py --run "$target" --robot "$robot" --gpu 0 "$@"
    fi
  done
  for name in "${names[@]}"; do
    if [[ $(docker wait "$name") != 0 ]]; then
      docker logs --tail 30 "$name" >&2
      status=1
    fi
  done
  return "$status"
}

(
  evaluate evaluate "$run"
  if [[ -n "$baseline" ]]; then
    if docker inspect "$scope-baseline-precompute" >/dev/null 2>&1; then
      docker wait "$scope-baseline-precompute" >/dev/null
    fi
    evaluate baseline "$baseline" --tracking-only --output "$run/evaluation/baseline"
  fi
) &
evaluation_pid=$!

if [[ ! -s "$run/feedback-verification.json" ]]; then
  docker run --rm --network none --gpus device=1 "${warp[@]}" "${runtime[@]}" "$train_image" \
    /entry/verify_mimic_feedback.py --run "$run"
fi

candidate=$scope-preview-staging
if docker inspect "$candidate" >/dev/null 2>&1; then docker rm -f "$candidate"; fi
docker run -d --name "$candidate" "${preview_env[@]}" "${runtime[@]}" "$preview_image" \
  /entry/serve_mimic_preview.py --run "$run" --device cuda:0 --port 32777
trap 'docker rm -f "$candidate" >/dev/null 2>&1 || true' EXIT
python3 "$run/tools/smoke_mimic_preview.py" http://127.0.0.1:32777 --iteration 10000
docker rm -f "$candidate"
wait "$evaluation_pid"

old=$public-before-global-ref
if docker inspect "$public" >/dev/null 2>&1; then
  if ! docker inspect "$old" >/dev/null 2>&1; then
    docker rename "$public" "$old"
    docker stop "$old"
  else
    docker rm -f "$public"
  fi
fi
if ! docker run -d --name "$public" --restart unless-stopped "${preview_env[@]}" "${runtime[@]}" "$preview_image" \
     /entry/serve_mimic_preview.py --run "$run" --device cuda:0 --host "$preview_host" ||
   ! python3 "$run/tools/smoke_mimic_preview.py" "http://$preview_host:31777" --iteration 10000 --ready-only; then
  docker rm -f "$public" >/dev/null 2>&1 || true
  if docker inspect "$old" >/dev/null 2>&1; then
    docker rename "$old" "$public"
    docker start "$public"
  fi
  exit 1
fi
