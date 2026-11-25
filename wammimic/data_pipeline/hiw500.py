"""Convert selected BitRobot/HIW-500-LeRobot G1 open-world episodes into unified WAM 5-point + gripper format.

Produces:
  - wam_lerobot_v2/g1_hiw500/data/chunk-000/episode_XXXXXX.parquet
      * action: [10, 47] (10x45 five-point in full t0 3D pelvis frame, leg_length=0.68m + 2D gripper targets)
      * observation.state_5pt_hist: [4, 79] (4x75 causal five-point pose/linvel/angvel + 4x4 [ee_meas(2), ee_res(2)])
      * anchor_quaternion_wxyz: [4]
      * valid_for_training: bool
  - wam_lerobot_v2/g1_hiw500/videos/chunk-000/observation.images.cam_head/episode_XXXXXX.mp4 (224x224 @ 10 Hz)
  - meta/{info.json, episodes.jsonl, tasks.jsonl, stats.json, splits.json, embodiment.json, modality.json}
"""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess

import mujoco
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

from .canonical_human import (
    CANONICAL_SCHEMA,
    ROBOT_GEOMETRY,
    compute_motion_metrics,
    measure_standing_leg_height,
    world_flu_to_canonical_5pt,
    world_flu_to_canonical_history,
)
from .common import sha256, write_json
from .openhlm import POINTS, SDK_JOINTS

NAME = "g1_hiw500"
DATASET = "wam_lerobot_v2/" + NAME
PROVENANCE = "hiw500_g1_fk_with_wbc_stance_odometry"
LEG_LENGTH = ROBOT_GEOMETRY["g1"]["leg_height"]
ARM_LENGTH = ROBOT_GEOMETRY["g1"]["arm_length"]


def _fk_sequence(xml_path: str, q29: np.ndarray, wbc23: np.ndarray) -> np.ndarray:
    """Compute 5-point world poses [N, 5, 7] (x,y,z, qw,qx,qy,qz) from G1 29-DoF joints + WBC pelvis roll/pitch/yaw."""
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in POINTS]
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in SDK_JOINTS]
    addresses = model.jnt_qposadr[joint_ids]
    sole_geoms = [
        i for i in range(model.ngeom)
        if model.geom_bodyid[i] in body_ids[1:3]
        and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_SPHERE
        and model.geom_contype[i]
    ]
    # wbc23: [pivot_vx, pivot_vy, pivot_vyaw, pivot_roll, pivot_pitch, pivot_yaw, pivot_height, ...]
    rots = Rotation.from_euler("xyz", wbc23[:, 3:6].astype(np.float64)).as_quat()  # xyzw
    result = np.empty((len(q29), 5, 7), dtype=np.float32)
    root = np.zeros(3, dtype=np.float64)
    prev = None
    for i in range(len(q29)):
        data.qpos[:3] = 0.0
        q_xyzw = rots[i]
        data.qpos[3:7] = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
        data.qpos[addresses] = q29[i]
        mujoco.mj_kinematics(model, data)
        pos = data.xpos[body_ids].copy()
        sole_z = data.geom_xpos[sole_geoms, 2] - model.geom_size[sole_geoms, 0]
        if prev is not None:
            stance = 1 + int(pos[2, 2] < pos[1, 2])
            root[:2] += prev[stance, :2] - pos[stance, :2]
        root[2] = -float(sole_z.min())
        result[i, :, :3] = pos + root
        result[i, :, 3:] = data.xquat[body_ids]
        prev = pos
    return result


def _transcode_segment(src_mp4: Path, dst_mp4: Path, t_start: float, n_frames_10hz: int) -> int:
    dst_mp4.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst_mp4.with_name(dst_mp4.stem + ".tmp.mp4")
    # Stereo-safe monocular crop: for wide stereo streams (W/H > 2.2, e.g. 1280x480),
    # take the left eye center square (480x480 at x=80) to avoid the x=640 stereo seam.
    vf = "fps=10,crop=ih:ih:if(gt(iw/ih\\,2.2)\\,(iw/2-ih)/2\\,(iw-ih)/2):0,scale=224:224:flags=lanczos"
    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
        "-ss", f"{t_start:.4f}", "-i", str(src_mp4),
        "-vf", vf, "-frames:v", str(n_frames_10hz),
        "-an", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
        str(tmp),
    ]
    subprocess.run(cmd, check=True)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(tmp)],
        capture_output=True, text=True, check=True,
    )
    got = int(probe.stdout.strip())
    if got < n_frames_10hz - 3 or got > n_frames_10hz or got < 18:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"Frame count mismatch for {dst_mp4.name}: got {got} vs expected {n_frames_10hz}")
    tmp.replace(dst_mp4)
    return got


def _process_one_episode(args):
    try:
        return _process_one_episode_inner(args)
    except Exception as err:
        print(f"[SKIP ep_out={args[0]} src={args[1].get('episode_index')}] {type(err).__name__}: {str(err)[:160]}", flush=True)
        return None


def _process_one_episode_inner(args):
    ep_out_idx, ep_meta, raw_str, out_str, xml_str, task_to_id, split_name = args
    raw, out = Path(raw_str), Path(out_str)
    dataset = out / DATASET
    receipt = dataset / f"receipts/episode_{ep_out_idx:06d}.json"
    if receipt.is_file():
        return json.loads(receipt.read_text())

    dpath = raw / ep_meta["data_file"]
    vpath = raw / ep_meta["video_file"]
    ep_src_idx = int(ep_meta["episode_index"])

    tbl = pq.read_table(
        dpath,
        columns=["observation.state", "observation.state.wbc", "action", "timestamp", "episode_index"],
        filters=[("episode_index", "==", ep_src_idx)],
    )
    n30 = len(tbl)
    if n30 < 54:  # Need >= 18 frames @ 10 Hz (4 history + 10 future + 2 visual tail)
        return None

    q29 = np.asarray(tbl["observation.state"].to_pylist(), dtype=np.float32)
    wbc = np.asarray(tbl["observation.state.wbc"].to_pylist(), dtype=np.float32)
    act = np.asarray(tbl["action"].to_pylist(), dtype=np.float32)
    ts = tbl["timestamp"].to_numpy().astype(np.float64)
    if not (np.isfinite(q29).all() and np.isfinite(wbc).all() and np.isfinite(act).all()):
        return None

    # Downsample 30 Hz -> 10 Hz by exact stride 3
    idx10 = np.arange(0, n30, 3, dtype=np.int64)
    n10 = len(idx10)
    if n10 < 18:
        return None

    video_dst = dataset / f"videos/chunk-000/observation.images.cam_head/episode_{ep_out_idx:06d}.mp4"
    got_n10 = _transcode_segment(vpath, video_dst, float(ep_meta["from_timestamp"]), n10)
    if got_n10 < n10:
        n10 = got_n10
        idx10 = idx10[:n10]

    poses30 = _fk_sequence(xml_str, q29, wbc)
    ts30 = np.arange(n30, dtype=np.float64) / 30.0
    poses10 = poses30[idx10]
    ts10 = np.arange(n10, dtype=np.float64) * 0.1
    try:
        leg_h = measure_standing_leg_height(poses30)
    except ValueError:
        leg_h = LEG_LENGTH
    arm_l = ARM_LENGTH

    # Gripper channels from wbc / action: [left_trigger(19), right_trigger(21)] clipped to [0, 1]
    ee_meas = np.clip(wbc[idx10][:, [19, 21]], 0.0, 1.0).astype(np.float32)
    ee_cmd = np.clip(act[idx10][:, [19, 21]], 0.0, 1.0).astype(np.float32)
    ee_res = (ee_cmd - ee_meas).astype(np.float32)
    ee_obs4 = np.concatenate((ee_meas, ee_res), axis=-1)  # [n10, 4]

    valid = np.zeros(n10, dtype=bool)
    valid[4 : n10 - 12] = True
    frames = np.flatnonzero(valid)

    actions = np.zeros((n10, 10, 47), dtype=np.float32)
    states = np.zeros((n10, 4, 79), dtype=np.float32)
    anchors_q = poses10[:, 0, 3:].astype(np.float32)

    for i in frames:
        actions[i, :, :45] = world_flu_to_canonical_5pt(poses10[i + 1 : i + 11], poses10[i], leg_h, arm_l)
        actions[i, :, 45:47] = ee_cmd[i + 1 : i + 11]

    hist_idx = frames[:, None] + np.arange(-3, 1)
    states[frames, :, :75] = world_flu_to_canonical_history(poses30, ts30, idx10[hist_idx], leg_h, arm_l)
    states[frames, :, 75:79] = ee_obs4[hist_idx]

    p_disp = np.zeros(n10, dtype=np.float32)
    a_lift = np.zeros(n10, dtype=np.float32)
    w_disp = np.zeros(n10, dtype=np.float32)
    buckets = np.full(n10, 2, dtype=np.int8)
    if len(frames) > 0:
        pd_v, al_v, wd_v, bk_v = compute_motion_metrics(actions[frames, :, :45], states[frames, -1, :45])
        p_disp[frames], a_lift[frames], w_disp[frames], buckets[frames] = pd_v, al_v, wd_v, bk_v

    task_id = int(task_to_id[ep_meta["task"]])
    table = pa.table({
        "action": pa.array(actions.tolist(), type=pa.list_(pa.list_(pa.float32(), 47), 10)),
        "observation.state_5pt_hist": pa.array(states.tolist(), type=pa.list_(pa.list_(pa.float32(), 79), 4)),
        "anchor_quaternion_wxyz": pa.array(anchors_q.tolist(), type=pa.list_(pa.float32(), 4)),
        "valid_for_training": valid,
        "timestamp": ts10.astype(np.float32),
        "frame_index": np.arange(n10, dtype=np.int64),
        "task_index": np.full(n10, task_id, dtype=np.int64),
        "episode_index": np.full(n10, ep_out_idx, dtype=np.int64),
        "pelvis_disp_1s": pa.array(p_disp, type=pa.float32()),
        "ankle_max_lift_1s": pa.array(a_lift, type=pa.float32()),
        "wrist_disp_1s": pa.array(w_disp, type=pa.float32()),
        "motion_bucket": pa.array(buckets, type=pa.int8()),
    })
    parquet_dst = dataset / f"data/chunk-000/episode_{ep_out_idx:06d}.parquet"
    parquet_dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, parquet_dst, compression="zstd")

    row = {
        "episode_index": ep_out_idx,
        "source_episode_index": ep_src_idx,
        "task": ep_meta["task"],
        "task_index": task_id,
        "split": split_name,
        "length": n10,
        "valid_samples": int(len(frames)),
        "canonical_leg_height_m": round(leg_h, 5),
        "canonical_arm_length_m": round(arm_l, 5),
        "bucket_counts": {
            "locomotion": int((buckets[frames] == 0).sum()),
            "manipulation": int((buckets[frames] == 1).sum()),
            "idle": int((buckets[frames] == 2).sum()),
        },
        "outputs": {
            str(video_dst.relative_to(out)): sha256(video_dst),
            str(parquet_dst.relative_to(out)): sha256(parquet_dst),
        },
    }
    receipt.parent.mkdir(parents=True, exist_ok=True)
    write_json(receipt, row)
    return row


def import_hiw500(raw_root: Path, output: Path, g1_xml: Path, workers: int = 12) -> dict:
    raw, out = Path(raw_root), Path(output)
    sel = json.loads((raw / "hiw500-100g-selection.json").read_text())
    dataset = out / DATASET
    dataset.mkdir(parents=True, exist_ok=True)

    identity = {
        "schema": "hiw500-g1-five-point-gripper-v2",
        "canonical_version": CANONICAL_SCHEMA,
        "subject_kind": "robot",
        "robot": "g1",
        "embodiment_tag": "bimanual_parallel_gripper",
        "label_provenance": PROVENANCE,
        "source_repo": sel["repo_id"],
        "source_sha": sel["sha"],
        "observation_state_5pt_hist": {"shape": [4, 79], "rate_hz": 10},
        "coordinate_frame": "Canonical Human Z-up FLU, Yaw-Only {P_t0}, dual-scale H_leg/L_arm",
        "video_crop": "left_eye_center_480x480_stereo_safe",
    }
    write_json(dataset / "source.json", identity)

    tasks_sorted = sorted({e["task"] for e in sel["episodes"]})
    task_to_id = {t: i for i, t in enumerate(tasks_sorted)}

    # Deterministic per-task 80/10/10 episode split
    by_task = {}
    for e in sel["episodes"]:
        by_task.setdefault(e["task"], []).append(e)
    jobs = []
    out_idx = 0
    for task in tasks_sorted:
        eps = sorted(by_task[task], key=lambda x: x["episode_index"])
        n = len(eps)
        n_val = max(1, round(0.1 * n)) if n >= 3 else 0
        n_test = max(1, round(0.1 * n)) if n >= 3 else 0
        n_train = n - n_val - n_test
        for i, ep_meta in enumerate(eps):
            split_name = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
            jobs.append((out_idx, ep_meta, str(raw), str(out), str(g1_xml), task_to_id, split_name))
            out_idx += 1

    episodes = []
    skipped = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for res in pool.map(_process_one_episode, jobs):
            if res is None:
                skipped += 1
            else:
                episodes.append(res)
                if len(episodes) % 100 == 0 or len(episodes) == len(jobs) - skipped:
                    print(f"HIW-500 prepared {len(episodes)}/{len(jobs)} episodes (skipped={skipped})", flush=True)

    meta = dataset / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in episodes))
    (meta / "tasks.jsonl").write_text("".join(json.dumps({"task_index": i, "task": t}) + "\n" for t, i in task_to_id.items()))

    # Compute train-only normalization statistics for action [10, 47] and observation.state_5pt_hist [4, 79]
    statistics = {}
    for key, width in [("action", 47), ("observation.state_5pt_hist", 79)]:
        arrays = []
        for row in episodes:
            if row["split"] != "train":
                continue
            tbl = pq.read_table(dataset / f"data/chunk-000/episode_{row['episode_index']:06d}.parquet", columns=[key, "valid_for_training"])
            mask = tbl["valid_for_training"].to_numpy()
            arr = tbl[key].combine_chunks().flatten().flatten().to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
            n_rows = len(mask)
            arr = arr.reshape(n_rows, -1, width)[mask].reshape(-1, width)
            arrays.append(arr)
        cat = np.concatenate(arrays, axis=0)
        statistics[key] = {
            "count": int(len(cat)),
            "mean": cat.mean(0).tolist(),
            "std": cat.std(0).tolist(),
            "q01": np.quantile(cat, 0.01, axis=0).tolist(),
            "q99": np.quantile(cat, 0.99, axis=0).tolist(),
        }

    write_json(meta / "stats.json", statistics)
    write_json(meta / "splits.json", {s: [r["episode_index"] for r in episodes if r["split"] == s] for s in ("train", "val", "test")})
    write_json(meta / "info.json", {
        **identity,
        "codebase_version": "v2.1",
        "world_model": "Cosmos-Predict2.5-2B",
        "fps": 10,
        "total_episodes": len(episodes),
        "total_frames": sum(r["length"] for r in episodes),
        "total_tasks": len(task_to_id),
        "features": {
            "action": {"dtype": "float32", "shape": [10, 47]},
            "observation.state_5pt_hist": {"dtype": "float32", "shape": [4, 79]},
            "observation.images.cam_head": {"dtype": "video", "shape": [3, 224, 224]},
        },
    })
    write_json(meta / "embodiment.json", {"embodiment_tag": "bimanual_parallel_gripper", "label_provenance": PROVENANCE, "end_effector_supervision": True})
    write_json(meta / "modality.json", {"action": {"five_point": [0, 45], "ee": [45, 47]}, "action_horizon": 10, "history_frames": 4})
    report = {
        "episodes": len(episodes),
        "skipped": skipped,
        "valid_windows": sum(r["valid_samples"] for r in episodes),
        "split_windows": {s: sum(r["valid_samples"] for r in episodes if r["split"] == s) for s in ("train", "val", "test")},
        "label_provenance": PROVENANCE,
    }
    write_json(out / "qa_reports/hiw500_import.json", report)
    return report
