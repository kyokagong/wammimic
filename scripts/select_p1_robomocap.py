"""Select a pinned, participant-balanced optical supplement before downloading it."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import requests

from scripts.download_cache_models import verify

REPO = "BitRobot/robomocap"
REVISION = "20adb80c474c163fe451b5c4f994b665b3234024"
FAMILIES = ("static_manipulation", "locomanipulation",
            "environment_coupled_wholebody", "long_horizon_composite")
FILES = ("annotations.json", "meta.json", "motions/skeleton.npz", "raw/manifest.json",
         "raw/mocap/Body0.bvh", "videos/frame_timestamps.npz", "videos/head_left_front.mp4")


def request(method, url, **kwargs):
    for attempt in range(5):
        try:
            response = requests.request(method, url, timeout=(20, 60), **kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt == 4:
                raise
            time.sleep(2 * (attempt + 1))


def balanced(rows, count):
    """Cover participants first, then primitives; duration is only a tie breaker."""
    people, primitives = Counter(), Counter()
    remaining = list(rows)
    chosen = []
    while remaining and len(chosen) < count:
        row = min(remaining, key=lambda r: (people[r["participant"]], primitives[r["primitive"]],
                                          r["duration_s"], r["release_path"]))
        chosen.append(row)
        remaining.remove(row)
        people[row["participant"]] += 1
        primitives[row["primitive"]] += 1
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=Path, required=True)
    parser.add_argument("--existing-manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--control", type=Path, required=True)
    args = parser.parse_args()
    args.control.mkdir(parents=True, exist_ok=True)
    rows = json.loads(args.sessions.read_text())["rows"]
    used = {r["source_session"] for p in args.existing_manifest
            for r in json.loads(p.read_text())["sessions"]}
    pool = [r for r in rows if r["family"] in FAMILIES and r["source_session"] not in used
            and r["has_annotation"] and r["skeleton_valid"] >= .8
            and r["sync_peak_corr"] >= .95 and "head_left_front" in r["videos"]]
    candidates = []
    for family in FAMILIES:
        for split, count in (("train", 150), ("val", 15), ("test", 15)):
            candidates.extend(balanced([r for r in pool if r["family"] == family and r["split"] == split], count))

    def inspect(row):
        path = args.control / "session-meta" / row["release_path"] / "meta.json"
        if not path.exists():
            response = request("GET", f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{row['release_path']}/meta.json")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(response.content)
        meta = json.loads(path.read_text())
        view = meta["streams"]["videos"]["left_front"]
        accepted = (meta["alignment"]["windows"]["lag_scatter_ms"] <= 5
                    and view["calibrated"] and not view["pts_gaps"] and view["fps"] == 30)
        return row, accepted, meta["alignment"]["windows"]["lag_scatter_ms"]

    with ThreadPoolExecutor(max_workers=8) as executor:
        inspected = list(executor.map(inspect, candidates))
    qualified = [r for r, ok, _ in inspected if ok]
    selected = []
    for family in FAMILIES:
        for split, count in (("train", 50), ("val", 6), ("test", 6)):
            selected.extend(balanced([r for r in qualified if r["family"] == family and r["split"] == split], count))
    # Deduplicate entire source recordings, rather than just release-path aliases.
    unique = {}
    for row in selected:
        old = unique.setdefault(row["source_session"], row)
        if old["split"] != row["split"]:
            raise ValueError("Recording appears in different official splits")
    selected = list(unique.values())
    paths = [row["release_path"] + "/" + name for row in selected for name in FILES]
    paths += ["README.md", "metadata/definitions/skeleton.json"]
    paths += [f"metadata/splits/{s}.txt" for s in ("train", "val", "test")]
    siblings = []
    for start in range(0, len(paths), 100):
        records = request("POST", f"https://huggingface.co/api/datasets/{REPO}/paths-info/{REVISION}",
                          json={"paths": paths[start:start + 100]}).json()
        for record in records:
            if record["type"] != "file":
                raise ValueError("Expected an explicit source file")
            entry = {"rfilename": record["path"], "blobId": record["oid"], "size": record["size"]}
            if record.get("lfs"):
                entry["lfs"] = {"sha256": record["lfs"]["oid"], "size": record["size"]}
            siblings.append(entry)
    present = {r["rfilename"] for r in siblings}
    missing = sorted(set(paths) - present)
    selected = [row for row in selected if all(row["release_path"] + "/" + name in present for name in FILES)]
    paths = {row["release_path"] + "/" + name for row in selected for name in FILES}
    paths.update(p for p in present if not p.startswith("actions/"))
    siblings = [r for r in siblings if r["rfilename"] in paths]
    (args.control / "robomocap-missing-source-files.json").write_text(json.dumps(missing, indent=2) + "\n")
    if not selected:
        raise ValueError(f"No complete source sessions: {missing[:10]}")
    # Metadata fetched for selection must match the pinned repository blobs too.
    for record in siblings:
        if record["rfilename"].endswith("/meta.json"):
            if not verify(args.control / "session-meta" / record["rfilename"],
                          {"size": record["size"], "blob_sha1": record["blobId"]}):
                raise ValueError("Selection metadata does not match the pinned source")
    manifest = {"id": REPO, "sha": REVISION, "siblings": siblings, "sessions": selected,
                "truth_policy": "optical_mocap_solved_skeleton; independent BVH verification still required",
                "view": "head_left_front", "excluded_sessions": [
                    {"source_session": r["release_path"], "lag_scatter_ms": lag}
                    for r, ok, lag in inspected if not ok]}
    summary = {family: {split: {"sessions": len(group),
                "participants": sorted({r["participant"] for r in group}),
                "duration_s": sum(r["duration_s"] for r in group)}
                for split in ("train", "val", "test")
                for group in [[r for r in selected if r["family"] == family and r["split"] == split]]}
               for family in FAMILIES}
    for name, data in (("robomocap-p1.json", manifest), ("robomocap-selection.json", summary)):
        (args.control / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"selected": len(selected), "bytes": sum(r["size"] for r in siblings),
                      "coverage": summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
