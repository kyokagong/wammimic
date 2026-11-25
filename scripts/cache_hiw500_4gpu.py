"""Parallel 4-GPU Cosmos-Predict2.5 VAE video latent caching for HIW-500 v2.0."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from wammimic.data_pipeline.cache import CORE_COMMIT, RECIPE, VAE_SHA256, cache_owner, cache_videos
from wammimic.data_pipeline.common import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--gpus", type=str, default="0,1,2,3")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dataset-name", type=str, default="g1_hiw500")
    args = parser.parse_args()

    gpu_ids = [g.strip() for g in args.gpus.split(",") if g.strip()]
    dataset = args.output_root / "wam_lerobot_v2" / args.dataset_name
    rows = [json.loads(l) for l in (dataset / "meta/episodes.jsonl").read_text().splitlines()]

    # Pre-create latents/source.json once to prevent multi-process race
    latent_root = dataset / "latents"
    latent_root.mkdir(parents=True, exist_ok=True)
    cache_owner(
        latent_root / "source.json",
        {
            "recipe": RECIPE,
            "vae_sha256": VAE_SHA256,
            "dataset_source_sha256": sha256(dataset / "source.json"),
            "source_commit": CORE_COMMIT,
        },
        list(latent_root.glob("chunk-*")),
    )

    # Greedy load-balanced sharding by valid_samples across GPUs
    pending = []
    for r in rows:
        ep = int(r["episode_index"])
        receipt = dataset / f"latents/chunk-000/episode_{ep:06d}.json"
        if not receipt.is_file():
            pending.append((int(r["valid_samples"]), ep))
    pending.sort(reverse=True)

    shards = [[] for _ in gpu_ids]
    loads = [0] * len(gpu_ids)
    for w, ep in pending:
        idx = min(range(len(gpu_ids)), key=lambda i: loads[i])
        shards[idx].append(ep)
        loads[idx] += w

    args.log_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Launching {len(gpu_ids)} GPU workers for {len(pending)}/{len(rows)} pending episodes "
        f"(loads={loads}, batch_size={args.batch_size})",
        flush=True,
    )

    t0 = time.time()
    procs = []
    for i, gpu in enumerate(gpu_ids):
        if not shards[i]:
            continue
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["VIDEO_CACHE_EPISODES"] = ",".join(str(e) for e in sorted(shards[i]))
        log_path = args.log_dir / f"hiw500-v2-cache-gpu{gpu}.log"
        log_f = log_path.open("w")
        cmd = [
            sys.executable,
            "-c",
            (
                "from wammimic.data_pipeline.cache import cache_videos; "
                f"print(cache_videos({str(args.model_root)!r}, {str(args.output_root)!r}, "
                f"batch_size={args.batch_size}, dataset_name={args.dataset_name!r}), flush=True)"
            ),
        ]
        p = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
        procs.append((gpu, p, log_f, log_path))

    failed = []
    for gpu, p, log_f, log_path in procs:
        rc = p.wait()
        log_f.close()
        print(f"GPU {gpu} worker exited with rc={rc} (log={log_path})", flush=True)
        if rc != 0:
            failed.append((gpu, rc, log_path))

    if failed:
        raise RuntimeError(f"GPU workers failed: {failed}")

    # Finalize unified qa_reports/cache_video.json (all receipts exist -> fast verification pass)
    report = cache_videos(args.model_root, args.output_root, batch_size=args.batch_size, dataset_name=args.dataset_name)
    print(
        f"ALL DONE in {(time.time() - t0) / 60:.1f} min | complete={report['complete']} "
        f"episodes={len(report['episodes'])} samples={report['samples']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
