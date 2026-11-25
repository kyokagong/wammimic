"""Refresh RoboMoCap, OpenHLM (gripper), and Nymeria-40 to Canonical Human v2.0.

Reuses 100% of existing 224x224 MP4 videos, Cosmos VAE video latents, and
Cosmos-Reason1 text embeddings. Rewrites only the 5-point action / proprio
parquet columns, adds motion diversity tags (pelvis_disp_1s, ankle_max_lift_1s,
wrist_disp_1s, motion_bucket), and refreshes checksum receipts cleanly.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from wammimic.data_pipeline.canonical_human import (
    CANONICAL_SCHEMA,
    CANON_ARM_LENGTH,
    CANON_LEG_HEIGHT,
    C_NOKOV_TO_FLU,
    ROBOT_GEOMETRY,
    compute_motion_metrics,
    measure_standing_leg_height,
    reproject_old_history_to_canonical,
    world_flu_to_canonical_5pt,
    world_flu_to_canonical_history,
)
from wammimic.data_pipeline.common import sha256, write_json


def _update_latent_receipt(dataset: Path, episode: int, parquet_sha: str) -> None:
    receipt = dataset / f"latents/chunk-000/episode_{episode:06d}.json"
    if receipt.is_file():
        data = json.loads(receipt.read_text())
        data["source_parquet_sha256"] = parquet_sha
        write_json(receipt, data)


def _sync_cache_video_report(branch_root: Path, dataset: Path, episodes: list[dict]) -> None:
    src_sha = sha256(dataset / "source.json")
    lat_src = dataset / "latents/source.json"
    if lat_src.is_file():
        ls = json.loads(lat_src.read_text())
        ls["dataset_source_sha256"] = src_sha
        write_json(lat_src, ls)
    cv_path = branch_root / "qa_reports/cache_video.json"
    if cv_path.is_file():
        cv = json.loads(cv_path.read_text())
        ep_out_sha = {}
        for r in episodes:
            for k, v in r.get("outputs", {}).items():
                if k.endswith(".parquet"):
                    ep_out_sha[r["episode_index"]] = v
        for row in cv.get("episodes", []):
            ep = row["episode_index"]
            if ep in ep_out_sha:
                row["source_parquet_sha256"] = ep_out_sha[ep]
        write_json(cv_path, cv)


def _replace_or_append_columns(table: pa.Table, new_cols: dict[str, pa.Array]) -> pa.Table:
    names = list(table.column_names)
    arrays = [table[c] for c in names]
    name_to_idx = {n: i for i, n in enumerate(names)}
    for k, arr in new_cols.items():
        if k in name_to_idx:
            arrays[name_to_idx[k]] = arr
        else:
            names.append(k)
            arrays.append(arr)
    return pa.table(dict(zip(names, arrays)))


def _refresh_human_episode(args):
    branch_root_str, raw_root_str, row = args
    branch_root = Path(branch_root_str)
    dataset = branch_root / "wam_lerobot_v2/human_ego"
    ep = int(row["episode_index"])
    stem = f"episode_{ep:06d}"

    lbl = np.load(dataset / f"labels/{stem}.npz", allow_pickle=False)
    poses_world = lbl["poses_world"]  # [N, 5, 7]
    old_leg = float(lbl["human_leg_length"])
    valid_pose = lbl["quality_valid"] if "quality_valid" in lbl.files else lbl["pose_valid"]

    # Measure standing leg height H_leg^(s) from world Z-up poses
    leg_h = measure_standing_leg_height(poses_world, valid_pose)

    # Measure arm length L_arm^(s) from raw skeleton.npz if available, else anthropometric ratio
    arm_l = None
    if raw_root_str:
        sk_path = Path(raw_root_str) / row["source_session"] / "motions/skeleton.npz"
        if sk_path.is_file():
            with np.load(sk_path, allow_pickle=False) as sk:
                names = sk["joint_names"].tolist()
                offs = sk["offsets_m"]
                if all(n in names for n in ("LeftForeArm", "LeftHand", "RightForeArm", "RightHand")):
                    al = np.linalg.norm(offs[names.index("LeftForeArm")]) + np.linalg.norm(offs[names.index("LeftHand")])
                    ar = np.linalg.norm(offs[names.index("RightForeArm")]) + np.linalg.norm(offs[names.index("RightHand")])
                    arm_l = float(0.5 * (al + ar))
    if arm_l is None or not (0.25 < arm_l < 0.90):
        arm_l = float(leg_h * (CANON_ARM_LENGTH / CANON_LEG_HEIGHT))

    pq_path = dataset / f"data/chunk-000/{stem}.parquet"
    tbl = pq.read_table(pq_path)
    n = len(tbl)
    valid = tbl["valid_for_training"].to_numpy()
    frames = np.flatnonzero(valid)

    old_st = tbl["observation.state"].combine_chunks().flatten().flatten().to_numpy(zero_copy_only=False).reshape(n, 4, 75)

    # Recompute actions [N, 10, 45] in Canonical Human Yaw-Only FLU frame
    actions = np.zeros((n, 10, 45), dtype=np.float32)
    for i in frames:
        actions[i] = world_flu_to_canonical_5pt(
            poses_world[i + 1 : i + 11], poses_world[i], leg_h, arm_l, c_rest=C_NOKOV_TO_FLU
        )

    # Reproject 120Hz history into Canonical Human Yaw-Only FLU frame
    states = reproject_old_history_to_canonical(
        poses_world, old_st, frames, old_leg, leg_h, arm_l, c_rest=C_NOKOV_TO_FLU
    )

    p_disp = np.zeros(n, dtype=np.float32)
    a_lift = np.zeros(n, dtype=np.float32)
    w_disp = np.zeros(n, dtype=np.float32)
    buckets = np.full(n, 2, dtype=np.int8)
    if len(frames) > 0:
        pd_v, al_v, wd_v, bk_v = compute_motion_metrics(actions[frames], states[frames, -1, :45])
        p_disp[frames], a_lift[frames], w_disp[frames], buckets[frames] = pd_v, al_v, wd_v, bk_v

    new_tbl = _replace_or_append_columns(
        tbl,
        {
            "action": pa.array(actions.tolist(), type=pa.list_(pa.list_(pa.float32(), 45), 10)),
            "observation.state": pa.array(states.tolist(), type=pa.list_(pa.list_(pa.float32(), 75), 4)),
            "pelvis_disp_1s": pa.array(p_disp, type=pa.float32()),
            "ankle_max_lift_1s": pa.array(a_lift, type=pa.float32()),
            "wrist_disp_1s": pa.array(w_disp, type=pa.float32()),
            "motion_bucket": pa.array(buckets, type=pa.int8()),
        },
    )
    tmp = pq_path.with_suffix(".parquet.tmp")
    pq.write_table(new_tbl, tmp, compression="zstd")
    os.replace(tmp, pq_path)

    new_sha = sha256(pq_path)
    rel_pq = str(pq_path.relative_to(branch_root))
    row = dict(row)
    row["outputs"] = dict(row.get("outputs", {}))
    row["outputs"][rel_pq] = new_sha
    row["canonical_leg_height_m"] = round(leg_h, 5)
    row["canonical_arm_length_m"] = round(arm_l, 5)
    row["bucket_counts"] = {
        "locomotion": int((buckets[frames] == 0).sum()),
        "manipulation": int((buckets[frames] == 1).sum()),
        "idle": int((buckets[frames] == 2).sum()),
    }
    write_json(dataset / f"receipts/{stem}.json", row)
    _update_latent_receipt(dataset, ep, new_sha)
    return row


def refresh_robomocap(branch_root: Path, raw_root: Path | None, workers: int) -> dict:
    t0 = time.time()
    dataset = branch_root / "wam_lerobot_v2/human_ego"
    rows = [json.loads(l) for l in (dataset / "meta/episodes.jsonl").read_text().splitlines()]
    raw_str = str(raw_root) if raw_root and raw_root.is_dir() else ""
    jobs = [(str(branch_root), raw_str, r) for r in rows]
    updated = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for res in pool.map(_refresh_human_episode, jobs):
            updated.append(res)
    updated.sort(key=lambda r: r["episode_index"])
    (dataset / "meta/episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in updated))

    src = json.loads((dataset / "source.json").read_text())
    src["canonical_version"] = CANONICAL_SCHEMA
    src["coordinate_frame"] = "Canonical Human Z-up FLU, Yaw-Only {P_t0}, dual-scale H_leg/L_arm"
    src["local_axis_calibration"] = "C_NOKOV_TO_FLU"
    write_json(dataset / "source.json", src)

    info = json.loads((dataset / "meta/info.json").read_text())
    info["canonical_version"] = CANONICAL_SCHEMA
    info["coordinate_frame"] = src["coordinate_frame"]
    write_json(dataset / "meta/info.json", info)

    _sync_cache_video_report(branch_root, dataset, updated)
    b_tot = {k: sum(r["bucket_counts"][k] for r in updated) for k in ("locomotion", "manipulation", "idle")}
    print(f"[{branch_root.name}] Refreshed {len(updated)} episodes in {time.time()-t0:.1f}s | buckets={b_tot}", flush=True)
    return {"branch": branch_root.name, "episodes": len(updated), "buckets": b_tot}


def _refresh_gripper_episode(args):
    branch_root_str, row = args
    branch_root = Path(branch_root_str)
    dataset = branch_root / "wam_lerobot_v2/g1_parallel_gripper"
    ep = int(row["episode_index"])
    stem = f"episode_{ep:06d}"

    poses50 = np.load(branch_root / f"morph_5pt_store/g1/openhlm_{ep:06d}/poses.npy")  # [N50, 5, 7]
    ts50 = np.arange(len(poses50), dtype=np.float64) / 50.0
    leg_h = measure_standing_leg_height(poses50)
    arm_l = ROBOT_GEOMETRY["g1"]["arm_length"]

    pq_path = dataset / f"data/chunk-000/{stem}.parquet"
    tbl = pq.read_table(pq_path)
    n = len(tbl)
    idx10_in_50 = np.arange(n, dtype=np.int64) * 5
    poses10 = poses50[idx10_in_50]

    valid = tbl["valid_for_training"].to_numpy()
    frames = np.flatnonzero(valid)

    old_act = tbl["action"].combine_chunks().flatten().flatten().to_numpy(zero_copy_only=False).reshape(n, 10, 47).copy()
    old_st5 = tbl["observation.state_5pt_hist"].combine_chunks().flatten().flatten().to_numpy(zero_copy_only=False).reshape(n, 4, 79).copy()

    for i in frames:
        old_act[i, :, :45] = world_flu_to_canonical_5pt(poses10[i + 1 : i + 11], poses10[i], leg_h, arm_l)

    if len(frames) > 0:
        hist_idx_50 = idx10_in_50[frames[:, None] + np.arange(-3, 1)]
        old_st5[frames, :, :75] = world_flu_to_canonical_history(
            poses50, ts50, hist_idx_50, leg_h, arm_l, allow_zero_initial=True
        )

    p_disp = np.zeros(n, dtype=np.float32)
    a_lift = np.zeros(n, dtype=np.float32)
    w_disp = np.zeros(n, dtype=np.float32)
    buckets = np.full(n, 2, dtype=np.int8)
    if len(frames) > 0:
        pd_v, al_v, wd_v, bk_v = compute_motion_metrics(old_act[frames, :, :45], old_st5[frames, -1, :45])
        p_disp[frames], a_lift[frames], w_disp[frames], buckets[frames] = pd_v, al_v, wd_v, bk_v

    new_tbl = _replace_or_append_columns(
        tbl,
        {
            "action": pa.array(old_act.tolist(), type=pa.list_(pa.list_(pa.float32(), 47), 10)),
            "observation.state_5pt_hist": pa.array(old_st5.tolist(), type=pa.list_(pa.list_(pa.float32(), 79), 4)),
            "pelvis_disp_1s": pa.array(p_disp, type=pa.float32()),
            "ankle_max_lift_1s": pa.array(a_lift, type=pa.float32()),
            "wrist_disp_1s": pa.array(w_disp, type=pa.float32()),
            "motion_bucket": pa.array(buckets, type=pa.int8()),
        },
    )
    tmp = pq_path.with_suffix(".parquet.tmp")
    pq.write_table(new_tbl, tmp, compression="zstd")
    os.replace(tmp, pq_path)

    new_sha = sha256(pq_path)
    rel_pq = str(pq_path.relative_to(branch_root))
    row = dict(row)
    row["outputs"] = dict(row.get("outputs", {}))
    row["outputs"][rel_pq] = new_sha
    row["canonical_leg_height_m"] = round(leg_h, 5)
    row["canonical_arm_length_m"] = round(arm_l, 5)
    row["bucket_counts"] = {
        "locomotion": int((buckets[frames] == 0).sum()),
        "manipulation": int((buckets[frames] == 1).sum()),
        "idle": int((buckets[frames] == 2).sum()),
    }
    write_json(dataset / f"receipts/{stem}.json", row)
    _update_latent_receipt(dataset, ep, new_sha)
    return row


def refresh_gripper(branch_root: Path, workers: int) -> dict:
    t0 = time.time()
    dataset = branch_root / "wam_lerobot_v2/g1_parallel_gripper"
    rows = [json.loads(l) for l in (dataset / "meta/episodes.jsonl").read_text().splitlines()]
    jobs = [(str(branch_root), r) for r in rows]
    updated = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for res in pool.map(_refresh_gripper_episode, jobs):
            updated.append(res)
    updated.sort(key=lambda r: r["episode_index"])
    (dataset / "meta/episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in updated))

    src = json.loads((dataset / "source.json").read_text())
    src["canonical_version"] = CANONICAL_SCHEMA
    src["coordinate_frame"] = "Canonical Human Z-up FLU, Yaw-Only {P_t0}, dual-scale H_leg/L_arm"
    write_json(dataset / "source.json", src)

    info = json.loads((dataset / "meta/info.json").read_text())
    info["canonical_version"] = CANONICAL_SCHEMA
    info["coordinate_frame"] = src["coordinate_frame"]
    write_json(dataset / "meta/info.json", info)

    _sync_cache_video_report(branch_root, dataset, updated)
    b_tot = {k: sum(r["bucket_counts"][k] for r in updated) for k in ("locomotion", "manipulation", "idle")}
    print(f"[gripper] Refreshed {len(updated)} episodes in {time.time()-t0:.1f}s | buckets={b_tot}", flush=True)
    return {"branch": "gripper", "episodes": len(updated), "buckets": b_tot}


def _refresh_nymeria_episode(args):
    branch_root_str, row = args
    branch_root = Path(branch_root_str)
    dataset = branch_root / "wam_lerobot_v2/nymeria_ego"
    ep = int(row["episode_index"])
    stem = f"episode_{ep:06d}"

    lbl = np.load(dataset / f"labels/{stem}.npz", allow_pickle=False)
    poses_world = lbl["poses_world"]  # [N, 5, 7]
    old_leg = float(lbl["human_leg_length"])
    valid_pose = lbl["valid"]

    # Read exact T-pose H_leg^(s) and L_arm^(s) from source_arrays/<seq>/xdata.npz
    seq = row.get("sequence") or row.get("source_sequence")
    xd_path = branch_root / "source_arrays" / seq / "xdata.npz"
    leg_h, arm_l = None, None
    if xd_path.is_file():
        with np.load(xd_path, allow_pickle=False) as xd:
            if "tpose_segment_tXYZ" in xd.files:
                tp = xd["tpose_segment_tXYZ"].reshape(23, 3)
                leg_h = float(tp[0, 2] - 0.5 * (tp[21, 2] + tp[17, 2]))
                al = np.linalg.norm(tp[13] - tp[12]) + np.linalg.norm(tp[14] - tp[13])
                ar = np.linalg.norm(tp[9] - tp[8]) + np.linalg.norm(tp[10] - tp[9])
                arm_l = float(0.5 * (al + ar))
    if leg_h is None or not (0.40 < leg_h < 1.30):
        leg_h = measure_standing_leg_height(poses_world, valid_pose)
    if arm_l is None or not (0.25 < arm_l < 0.90):
        arm_l = float(leg_h * (CANON_ARM_LENGTH / CANON_LEG_HEIGHT))

    pq_path = dataset / f"data/chunk-000/{stem}.parquet"
    tbl = pq.read_table(pq_path)
    n = len(tbl)
    valid = tbl["valid_for_training"].to_numpy()
    frames = np.flatnonzero(valid)

    old_st = tbl["observation.state"].combine_chunks().flatten().flatten().to_numpy(zero_copy_only=False).reshape(n, 4, 75)

    actions = np.zeros((n, 10, 45), dtype=np.float32)
    for i in frames:
        actions[i] = world_flu_to_canonical_5pt(poses_world[i + 1 : i + 11], poses_world[i], leg_h, arm_l)

    states = reproject_old_history_to_canonical(poses_world, old_st, frames, old_leg, leg_h, arm_l)

    p_disp = np.zeros(n, dtype=np.float32)
    a_lift = np.zeros(n, dtype=np.float32)
    w_disp = np.zeros(n, dtype=np.float32)
    buckets = np.full(n, 2, dtype=np.int8)
    if len(frames) > 0:
        pd_v, al_v, wd_v, bk_v = compute_motion_metrics(actions[frames], states[frames, -1, :45])
        p_disp[frames], a_lift[frames], w_disp[frames], buckets[frames] = pd_v, al_v, wd_v, bk_v

    new_tbl = _replace_or_append_columns(
        tbl,
        {
            "action": pa.array(actions.tolist(), type=pa.list_(pa.list_(pa.float32(), 45), 10)),
            "observation.state": pa.array(states.tolist(), type=pa.list_(pa.list_(pa.float32(), 75), 4)),
            "pelvis_disp_1s": pa.array(p_disp, type=pa.float32()),
            "ankle_max_lift_1s": pa.array(a_lift, type=pa.float32()),
            "wrist_disp_1s": pa.array(w_disp, type=pa.float32()),
            "motion_bucket": pa.array(buckets, type=pa.int8()),
        },
    )
    tmp = pq_path.with_suffix(".parquet.tmp")
    pq.write_table(new_tbl, tmp, compression="zstd")
    os.replace(tmp, pq_path)

    new_sha = sha256(pq_path)
    rel_pq = str(pq_path.relative_to(branch_root))
    row = dict(row)
    row["outputs"] = dict(row.get("outputs", {}))
    row["outputs"][rel_pq] = new_sha
    row["canonical_leg_height_m"] = round(leg_h, 5)
    row["canonical_arm_length_m"] = round(arm_l, 5)
    row["bucket_counts"] = {
        "locomotion": int((buckets[frames] == 0).sum()),
        "manipulation": int((buckets[frames] == 1).sum()),
        "idle": int((buckets[frames] == 2).sum()),
    }
    write_json(dataset / f"receipts/{stem}.json", row)
    _update_latent_receipt(dataset, ep, new_sha)
    return row


def refresh_nymeria40(branch_root: Path, workers: int) -> dict:
    t0 = time.time()
    dataset = branch_root / "wam_lerobot_v2/nymeria_ego"
    rows = [json.loads(l) for l in (dataset / "meta/episodes.jsonl").read_text().splitlines()]
    jobs = [(str(branch_root), r) for r in rows]
    updated = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for res in pool.map(_refresh_nymeria_episode, jobs):
            updated.append(res)
    updated.sort(key=lambda r: r["episode_index"])
    (dataset / "meta/episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in updated))

    src = json.loads((dataset / "source.json").read_text())
    src["canonical_version"] = CANONICAL_SCHEMA
    src["coordinate_frame"] = "Canonical Human Z-up FLU, Yaw-Only {P_t0}, dual-scale H_leg/L_arm"
    write_json(dataset / "source.json", src)

    info = json.loads((dataset / "meta/info.json").read_text())
    info["canonical_version"] = CANONICAL_SCHEMA
    info["coordinate_frame"] = src["coordinate_frame"]
    write_json(dataset / "meta/info.json", info)

    _sync_cache_video_report(branch_root, dataset, updated)
    b_tot = {k: sum(r["bucket_counts"][k] for r in updated) for k in ("locomotion", "manipulation", "idle")}
    print(f"[nymeria-40] Refreshed {len(updated)} episodes in {time.time()-t0:.1f}s | buckets={b_tot}", flush=True)
    return {"branch": "nymeria-40", "episodes": len(updated), "buckets": b_tot}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--branches", nargs="+", default=["human-base", "human-new", "human-operations", "gripper", "nymeria-40"])
    args = parser.parse_args()

    results = {}
    for b in args.branches:
        b_dir = args.processed_root / b
        if b == "human-base":
            raw = args.raw_root / "robomocap_raw" if args.raw_root else None
            results[b] = refresh_robomocap(b_dir, raw, args.workers)
        elif b == "human-new":
            results[b] = refresh_robomocap(b_dir, None, args.workers)
        elif b == "human-operations":
            raw = args.raw_root / "robomocap_operations_raw_20261118" if args.raw_root else None
            results[b] = refresh_robomocap(b_dir, raw, args.workers)
        elif b == "gripper":
            results[b] = refresh_gripper(b_dir, args.workers)
        elif b == "nymeria-40":
            results[b] = refresh_nymeria40(b_dir, args.workers)
        else:
            raise ValueError(f"Unknown branch: {b}")

    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
