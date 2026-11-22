"""Resumable, SHA1-verified multi-threaded downloader for selected Nymeria sequences.

Never logs or persists signed URLs. Reads links strictly from the private links JSON.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

REQUIRED_KEYS = (
    "LICENSE",
    "metadata_json",
    "narration_activity_summarization_csv",
    "narration_atomic_action_csv",
    "body_motion",
    "recording_head",
    "recording_head_data_data_vrs",
)


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            chunk = f.read(8 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def download_one_file(url: str, dest: Path, expected_size: int, expected_sha1: str | None) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")

    if dest.is_file() and (expected_size <= 0 or dest.stat().st_size == expected_size):
        if not expected_sha1 or sha1_file(dest) == expected_sha1:
            return {"file": dest.name, "status": "cached", "bytes": dest.stat().st_size}

    for attempt in range(1, 26):
        try:
            existing = part.stat().st_size if part.is_file() else 0
            if expected_size > 0 and existing > expected_size:
                part.unlink(missing_ok=True)
                existing = 0
            headers = {"User-Agent": "wammimic-nymeria-downloader/1.0"}
            mode = "wb"
            if existing > 0:
                headers["Range"] = f"bytes={existing}-"
                mode = "ab"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=90) as resp:
                status = getattr(resp, "status", 200)
                if existing > 0 and status != 206:
                    existing = 0
                    mode = "wb"
                with part.open(mode) as out:
                    while True:
                        buf = resp.read(2 * 1024 * 1024)
                        if not buf:
                            break
                        out.write(buf)
            actual_size = part.stat().st_size
            if expected_size > 0 and actual_size != expected_size:
                raise IOError(f"size mismatch {actual_size} != {expected_size}")
            if expected_sha1:
                actual_sha1 = sha1_file(part)
                if actual_sha1 != expected_sha1:
                    part.unlink(missing_ok=True)
                    raise IOError(f"sha1 mismatch {actual_sha1} != {expected_sha1}")
            part.rename(dest)
            return {"file": dest.name, "status": "downloaded", "bytes": actual_size}
        except Exception as err:
            if attempt == 25:
                return {"file": dest.name, "status": "error", "error": f"{type(err).__name__}: {str(err)[:160]}"}
            time.sleep(min(2 * attempt, 15))
    return {"file": dest.name, "status": "error", "error": "exhausted retries"}


def download_sequence(seq_id: str, files_meta: dict, out_root: Path) -> dict:
    t0 = time.time()
    seq_dir = out_root / seq_id
    seq_dir.mkdir(parents=True, exist_ok=True)
    results = []
    ok = True
    total_bytes = 0
    for key in REQUIRED_KEYS:
        info = files_meta.get(key)
        if not isinstance(info, dict) or "download_url" not in info:
            continue
        fname = info.get("filename") or f"Nymeria_v0.0_{seq_id}_{key}"
        res = download_one_file(
            info["download_url"],
            seq_dir / fname,
            int(info.get("file_size_bytes") or 0),
            info.get("sha1sum"),
        )
        results.append(res)
        total_bytes += int(res.get("bytes", 0))
        if res["status"] == "error":
            ok = False
    return {
        "sequence": seq_id,
        "ok": ok,
        "bytes": total_bytes,
        "elapsed_s": round(time.time() - t0, 1),
        "files": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--links-json", type=Path, required=True)
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--status-json", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    links_raw = json.loads(args.links_json.read_text())
    seqs_map = links_raw.get("sequences", links_raw)
    sel = json.loads(args.selection_json.read_text())
    seq_ids = [row["sequence"] for row in sel["sequences"]]

    args.out_root.mkdir(parents=True, exist_ok=True)
    completed = []
    failed = []
    started = time.time()

    def write_status():
        payload = {
            "total": len(seq_ids),
            "completed": len(completed),
            "failed": len(failed),
            "downloaded_gb": round(sum(r["bytes"] for r in completed) / 1e9, 2),
            "elapsed_min": round((time.time() - started) / 60.0, 2),
            "completed_sequences": [r["sequence"] for r in completed],
            "failed_sequences": failed,
        }
        tmp = args.status_json.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(args.status_json)

    write_status()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(download_sequence, sid, seqs_map[sid], args.out_root): sid for sid in seq_ids if sid in seqs_map}
        for fut in concurrent.futures.as_completed(futs):
            res = fut.result()
            if res["ok"]:
                completed.append(res)
                print(f"[OK {len(completed)}/{len(seq_ids)}] {res['sequence']} ({res['bytes']/1e9:.2f} GB in {res['elapsed_s']}s)", flush=True)
            else:
                failed.append(res)
                print(f"[FAIL] {res['sequence']} -> {[f for f in res['files'] if f['status']=='error']}", flush=True)
            write_status()

    print(f"Finished: {len(completed)}/{len(seq_ids)} ok, {len(failed)} failed.", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
