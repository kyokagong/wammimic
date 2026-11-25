"""Select a ~100 GB task-balanced subset of BitRobot/HIW-500-LeRobot (head camera + data only) and download it."""
import argparse
from collections import defaultdict
import concurrent.futures
import json
from pathlib import Path
import time

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "BitRobot/HIW-500-LeRobot"

PRIORITY_BUDGET_BYTES = {
    "clean up the room": 12.0e9,
    "clothes washing": 10.0e9,
    "setting the table": 10.0e9,
    "kitchen organization": 10.0e9,
    "picking trash to rubbish bin": 10.0e9,
    "restocking fridge": 9.0e9,
    "hang keys on a hook": 9.0e9,
    "move the pillow to the sofa from floor": 8.0e9,
    "sweep floor": 6.0e9,
    "hang hanger": 6.0e9,
    "building children table": 6.0e9,
}


def build_selection(out_dir: Path, budget_gb: float) -> dict:
    api = HfApi()
    info = api.dataset_info(REPO_ID, files_metadata=True)
    size_map = {s.rfilename: int(s.size or 0) for s in info.siblings}

    cols = [
        "episode_index", "tasks", "length",
        "data/chunk_index", "data/file_index",
        "videos/observation.images.head/chunk_index", "videos/observation.images.head/file_index",
        "videos/observation.images.head/from_timestamp", "videos/observation.images.head/to_timestamp",
    ]
    ep_rows = []
    for p in sorted((out_dir / "meta/episodes").rglob("*.parquet")):
        ep_rows.extend(pq.read_table(p, columns=cols).to_pylist())

    vid_to_eps = defaultdict(list)
    for r in ep_rows:
        task = r["tasks"][0] if isinstance(r["tasks"], list) and r["tasks"] else str(r["tasks"])
        vc = int(r["videos/observation.images.head/chunk_index"])
        vf_i = int(r["videos/observation.images.head/file_index"])
        dc = int(r["data/chunk_index"])
        df_i = int(r["data/file_index"])
        vfile = f"videos/observation.images.head/chunk-{vc:03d}/file-{vf_i:03d}.mp4"
        dfile = f"data/chunk-{dc:03d}/file-{df_i:03d}.parquet"
        if vfile in size_map and dfile in size_map:
            vid_to_eps[vfile].append({
                "episode_index": int(r["episode_index"]),
                "task": task,
                "length": int(r["length"]),
                "data_file": dfile,
                "video_file": vfile,
                "from_timestamp": float(r["videos/observation.images.head/from_timestamp"]),
                "to_timestamp": float(r["videos/observation.images.head/to_timestamp"]),
            })

    task_vids = defaultdict(list)
    for vf, eps in vid_to_eps.items():
        task_vids[eps[0]["task"]].append((vf, size_map[vf], eps))

    scale = budget_gb / 96.0
    budget_bytes = int(budget_gb * 1e9)
    selected_vids = set()
    selected_datas = set()
    selected_eps = []
    total_bytes = 0

    for task, base_b in PRIORITY_BUDGET_BYTES.items():
        target_b = base_b * scale
        t_bytes = 0
        for vf, vsz, eps in sorted(task_vids.get(task, []), key=lambda x: x[0]):
            needed_d = {e["data_file"] for e in eps} - selected_datas
            add_b = vsz + sum(size_map[df] for df in needed_d)
            if t_bytes + add_b > target_b or total_bytes + add_b > budget_bytes:
                if t_bytes >= 0.75 * target_b:
                    break
                if total_bytes + add_b > budget_bytes:
                    continue
            selected_vids.add(vf)
            selected_datas.update(needed_d)
            selected_eps.extend(eps)
            t_bytes += add_b
            total_bytes += add_b

    selected_files = sorted(selected_datas) + sorted(selected_vids)
    per_task = defaultdict(lambda: [0, 0])
    for e in selected_eps:
        per_task[e["task"]][0] += 1
        per_task[e["task"]][1] += e["length"]

    total_frames = sum(e["length"] for e in selected_eps)
    print(f"Selected {len(selected_eps)} episodes, {total_frames:,} frames (@30Hz = {total_frames/30/3600:.1f} h)", flush=True)
    print(f"Files: {len(selected_datas)} data parquets + {len(selected_vids)} head MP4s = {len(selected_files)} files ({total_bytes/1e9:.2f} GB)", flush=True)
    for t, (cnt, fr) in sorted(per_task.items(), key=lambda x: -x[1][0]):
        print(f"  {cnt:4d} eps ({fr/30/60:5.1f} min)  {t}", flush=True)

    manifest = {
        "repo_id": REPO_ID,
        "sha": info.sha,
        "budget_gb": budget_gb,
        "total_bytes": total_bytes,
        "total_episodes": len(selected_eps),
        "total_frames_30hz": total_frames,
        "per_task_episodes": {t: {"episodes": c, "frames_30hz": f} for t, (c, f) in sorted(per_task.items())},
        "data_files": sorted(selected_datas),
        "video_files": sorted(selected_vids),
        "files": [{"path": f, "size": size_map[f]} for f in selected_files],
        "episodes": selected_eps,
    }
    (out_dir / "hiw500-100g-selection.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def download_all(out_dir: Path, manifest: dict, workers: int) -> None:
    import subprocess
    files = manifest["files"]
    status_path = out_dir / "hiw500-download-status.json"
    started = time.time()
    done_files = 0
    done_bytes = 0
    failed = []
    sha = manifest["sha"]

    def _fetch(item):
        rel = item["path"]
        expected = int(item["size"])
        dest = out_dir / rel
        if dest.is_file() and dest.stat().st_size == expected:
            return (rel, expected, True, "cached")
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_suffix(dest.suffix + ".part")
        url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/{sha}/{rel}"
        for attempt in range(1, 10):
            try:
                res = subprocess.run(
                    ["curl", "-fsSL", "-C", "-", "--connect-timeout", "20", "--max-time", "600",
                     "--retry", "3", "--retry-delay", "3", "-o", str(part), url],
                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                )
                if res.returncode != 0:
                    raise IOError(res.stderr.strip()[-120:] or f"curl rc={res.returncode}")
                if part.is_file() and part.stat().st_size == expected:
                    part.replace(dest)
                    return (rel, expected, True, "downloaded")
                actual = part.stat().st_size if part.is_file() else 0
                if actual > expected:
                    part.unlink(missing_ok=True)
                raise IOError(f"size mismatch {actual} != {expected}")
            except Exception as err:
                if attempt == 9:
                    return (rel, 0, False, f"{type(err).__name__}: {str(err)[:120]}")
                time.sleep(min(3 * attempt, 15))
        return (rel, 0, False, "exhausted")

    def _write_status():
        payload = {
            "total_files": len(files),
            "completed_files": done_files,
            "failed_files": len(failed),
            "downloaded_gb": round(done_bytes / 1e9, 2),
            "target_gb": round(manifest["total_bytes"] / 1e9, 2),
            "elapsed_min": round((time.time() - started) / 60.0, 2),
            "failed": failed[:20],
        }
        tmp = status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(status_path)

    _write_status()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_fetch, item) for item in files]
        for fut in concurrent.futures.as_completed(futs):
            rel, sz, ok, msg = fut.result()
            if ok:
                done_files += 1
                done_bytes += sz
                if done_files % 15 == 0 or done_files == len(files):
                    print(f"[{done_files}/{len(files)}] {done_bytes/1e9:.2f}/{manifest['total_bytes']/1e9:.2f} GB ({(time.time()-started)/60:.1f} min)", flush=True)
            else:
                failed.append({"path": rel, "error": msg})
                print(f"[FAIL] {rel}: {msg}", flush=True)
            _write_status()

    print(f"HIW-500 download finished: {done_files}/{len(files)} ok, {len(failed)} failed.", flush=True)
    if failed:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--budget-gb", type=float, default=98.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--select-only", action="store_true")
    args = parser.parse_args()

    sel_path = args.out_dir / "hiw500-100g-selection.json"
    if sel_path.is_file():
        manifest = json.loads(sel_path.read_text())
        print(f"Loaded existing selection: {manifest['total_episodes']} eps, {manifest['total_bytes']/1e9:.2f} GB", flush=True)
    else:
        manifest = build_selection(args.out_dir, args.budget_gb)
    if not args.select_only:
        download_all(args.out_dir, manifest, args.workers)


if __name__ == "__main__":
    main()
