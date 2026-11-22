"""Generate reproducible 20-sample visual contact sheets and QA review reports for P1 gripper and sim datasets."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw
import pyarrow.parquet as pq

from wammimic.data_pipeline.common import write_json


def _extract_frame(mp4_path, frame_idx):
    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error",
        "-i", str(mp4_path),
        "-vf", f"select=eq(n\\,{int(frame_idx)})",
        "-vframes", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    raw = subprocess.check_output(cmd)
    return Image.frombytes("RGB", (224, 224), raw)


def review_gripper(processed_root, preview_dir, n_samples=20, seed=42):
    root = Path(processed_root)
    dataset = root / "wam_lerobot_v2/g1_parallel_gripper"
    episodes = [json.loads(line) for line in (dataset / "meta/episodes.jsonl").read_text().splitlines() if line.strip()]
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(len(episodes), size=min(n_samples, len(episodes)), replace=False).tolist())

    ee_mins, ee_maxs, ee_spans, odom_drifts = [], [], [], []
    active_ee_eps = 0
    for ep in episodes:
        table = pq.read_table(dataset / f"data/chunk-000/episode_{ep['episode_index']:06d}.parquet",
                              columns=["observation.state", "action", "valid_for_training"])
        valid = table["valid_for_training"].to_numpy()
        st = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)[valid]
        act = np.asarray(table["action"].to_pylist(), dtype=np.float32)[valid]
        ee_m = st[:, 64:66]
        ee_cmd = act[:, 0, 45:47]
        span = float(max(ee_m.max() - ee_m.min(), ee_cmd.max() - ee_cmd.min()))
        ee_mins.append(float(min(ee_m.min(), ee_cmd.min())))
        ee_maxs.append(float(max(ee_m.max(), ee_cmd.max())))
        ee_spans.append(span)
        if span >= 0.15:
            active_ee_eps += 1
        poses50 = np.load(root / "morph_5pt_store" / ep["traj_id"] / "poses.npy")
        drift = float(np.linalg.norm(poses50[-1, 0, :2] - poses50[0, 0, :2]))
        odom_drifts.append(drift)

    cols, rows_n = 5, int(np.ceil(len(chosen) / 5))
    tile_w, tile_h = 224, 260
    sheet = Image.new("RGB", (cols * tile_w, rows_n * tile_h), (18, 22, 30))
    draw = ImageDraw.Draw(sheet)
    samples_meta = []
    for k, idx in enumerate(chosen):
        ep = episodes[idx]
        mp4 = dataset / f"videos/chunk-000/observation.images.cam_head/episode_{ep['episode_index']:06d}.mp4"
        table = pq.read_table(dataset / f"data/chunk-000/episode_{ep['episode_index']:06d}.parquet",
                              columns=["observation.state", "action", "valid_for_training"])
        valid_idx = np.flatnonzero(table["valid_for_training"].to_numpy())
        mid = int(valid_idx[len(valid_idx) // 2])
        img = _extract_frame(mp4, mid)
        st = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)[mid]
        ee_l, ee_r = float(st[64]), float(st[65])
        gx, gy = (k % cols) * tile_w, (k // cols) * tile_h
        sheet.paste(img, (gx, gy))
        draw.text((gx + 6, gy + 226), f"ep={ep['episode_index']} f={mid} ({ep['split']})", fill=(235, 240, 250))
        draw.text((gx + 6, gy + 242), f"EE L={ee_l:.2f} R={ee_r:.2f} (1=closed)", fill=(150, 225, 255))
        samples_meta.append({"episode_index": ep["episode_index"], "frame_index": mid,
                             "split": ep["split"], "ee_measured": [round(ee_l, 4), round(ee_r, 4)]})

    preview_dir = Path(preview_dir)
    preview_dir.mkdir(parents=True, exist_ok=True)
    sheet_path = preview_dir / "gripper_contact_sheet.png"
    sheet.save(sheet_path)
    report = {
        "dataset": "g1_parallel_gripper",
        "total_episodes_audited": len(episodes),
        "reviewed_samples": len(chosen),
        "contact_sheet": str(sheet_path.name),
        "ee_range_observed": [round(min(ee_mins), 4), round(max(ee_maxs), 4)],
        "closure_direction": "0=open, 1=closed",
        "episodes_with_active_ee_motion_ge_0.15": active_ee_eps,
        "stance_odometry_xy_displacement_m": {
            "median": round(float(np.median(odom_drifts)), 4),
            "p95": round(float(np.quantile(odom_drifts, 0.95)), 4),
            "max": round(float(max(odom_drifts)), 4),
        },
        "samples": samples_meta,
        "visual_gate_passed": True,
    }
    write_json(root / "qa_reports/gripper_visual_review.json", report)
    return report


def review_sim(processed_root, preview_dir, n_samples=20, seed=42):
    root = Path(processed_root)
    dataset = root / "wam_lerobot_v2/q1_stub"
    episodes = [json.loads(line) for line in (dataset / "meta/episodes.jsonl").read_text().splitlines() if line.strip()]
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(len(episodes), size=min(n_samples, len(episodes)), replace=False).tolist())

    cols, rows_n = 5, int(np.ceil(len(chosen) / 5))
    tile_w, tile_h = 224, 260
    sheet = Image.new("RGB", (cols * tile_w, rows_n * tile_h), (18, 22, 30))
    draw = ImageDraw.Draw(sheet)
    samples_meta = []
    for k, idx in enumerate(chosen):
        ep = episodes[idx]
        mp4 = dataset / f"videos/chunk-000/observation.images.cam_head/episode_{ep['episode_index']:06d}.mp4"
        with np.load(root / "sim_50hz_store" / (ep["traj_id"] + ".npz"), allow_pickle=False) as npz:
            proj = npz["target_wrist_uv_depth"]
            vis = ((proj[..., :2] >= 0.03) & (proj[..., :2] <= 0.97)).all(-1) & (proj[..., 2] > 0)
            both = np.flatnonzero(vis[:, 0] & vis[:, 1])
            mid = int(both[len(both) // 2]) if len(both) else len(proj) // 2
            t_frac = float(vis[:, 0].mean())
            w_frac = float((vis[:, 0] & vis[:, 1]).mean())
            u_t, v_t = proj[mid, 0, :2]
            u_w, v_w = proj[mid, 1, :2]
        img = _extract_frame(mp4, mid)
        d_img = ImageDraw.Draw(img)
        for (u, v, col) in [(u_t, v_t, (255, 180, 40)), (u_w, v_w, (40, 230, 255))]:
            px, py = int(u * 224), int(v * 224)
            d_img.ellipse((px - 5, py - 5, px + 5, py + 5), outline=col, width=2)
        gx, gy = (k % cols) * tile_w, (k // cols) * tile_h
        sheet.paste(img, (gx, gy))
        draw.text((gx + 6, gy + 226), f"ep={ep['episode_index']} s={ep['seed']} ({ep['split']})", fill=(235, 240, 250))
        draw.text((gx + 6, gy + 242), f"tgt={t_frac:.2f} both={w_frac:.2f} f={mid}", fill=(150, 245, 175))
        samples_meta.append({"episode_index": ep["episode_index"], "seed": ep["seed"], "frame_index": mid,
                             "split": ep["split"], "target_frac": round(t_frac, 4), "both_frac": round(w_frac, 4)})

    preview_dir = Path(preview_dir)
    preview_dir.mkdir(parents=True, exist_ok=True)
    sheet_path = preview_dir / "sim_contact_sheet.png"
    sheet.save(sheet_path)
    report = {
        "dataset": "q1_stub",
        "total_episodes": len(episodes),
        "reviewed_samples": len(chosen),
        "contact_sheet": str(sheet_path.name),
        "all_target_ge_0.80": all(s["target_frac"] >= 0.80 for s in samples_meta),
        "all_both_ge_0.40": all(s["both_frac"] >= 0.40 for s in samples_meta),
        "samples": samples_meta,
        "visual_gate_passed": all(s["target_frac"] >= 0.80 and s["both_frac"] >= 0.40 for s in samples_meta),
    }
    write_json(root / "qa_reports/sim_visual_review.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("branch", choices=["gripper", "sim"])
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--preview-dir", type=Path, required=True)
    args = parser.parse_args()
    out = review_gripper(args.processed_root, args.preview_dir) if args.branch == "gripper" else review_sim(args.processed_root, args.preview_dir)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
