"""Render synchronized Ego-RGB + GT vs WAM v2.0 Predicted Canonical Human 5-point 3D videos."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pyarrow.parquet as pq
from safetensors import safe_open
import torch

from wammimic.data_pipeline.canonical_human import CANON_LEG_HEIGHT
from wammimic.data_pipeline.geometry import rotation6d
from wammimic.training.evaluation import autocast
from wammimic.training.model import CosmosWAM, load_backbone


POINT_NAMES = ("Pelvis", "L-Ankle", "R-Ankle", "L-Wrist", "R-Wrist")
POINT_COLORS = (
    (255, 140, 40),   # Pelvis: orange
    (255, 210, 60),   # L-Ankle: gold
    (240, 100, 90),   # R-Ankle: coral
    (40, 220, 255),   # L-Wrist: cyan
    (235, 95, 235),   # R-Wrist: magenta
)
BUCKET_META = {
    0: ("Bucket A · Locomotion", (46, 170, 95)),
    1: ("Bucket B · Manipulation", (36, 155, 215)),
    2: ("Bucket C · Idle", (110, 122, 138)),
}


def decode_video_rgb(video_path: Path, width: int = 224, height: int = 224) -> np.ndarray:
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", str(video_path),
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "pipe:1",
    ]
    raw = subprocess.check_output(cmd)
    frame_bytes = width * height * 3
    n_frames = len(raw) // frame_bytes
    return np.frombuffer(raw[: n_frames * frame_bytes], dtype=np.uint8).reshape(n_frames, height, width, 3)


def make_camera(eye: np.ndarray, target: np.ndarray):
    fwd = target - eye
    fwd = fwd / np.linalg.norm(fwd)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(fwd, world_up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, fwd)
    return eye, right, up, fwd


def project_points(pts: np.ndarray, cam, rect: tuple[int, int, int, int], focal: float = 520.0):
    eye, right, up, fwd = cam
    x0, y0, w, h = rect
    cx, cy = x0 + w * 0.5, y0 + h * 0.54
    rel = pts - eye
    z = rel @ fwd
    z_safe = np.maximum(z, 0.1)
    u = cx + focal * (rel @ right) / z_safe
    v = cy - focal * (rel @ up) / z_safe
    return np.stack([u, v], axis=-1), z


def canonical_45_to_meters_and_rots(feat_45: np.ndarray):
    pos = feat_45[..., :15].reshape(feat_45.shape[:-1] + (5, 3)).astype(np.float64).copy()
    pos[..., :3, :] *= CANON_LEG_HEIGHT
    r6 = feat_45[..., 15:45].reshape(feat_45.shape[:-1] + (5, 6)).astype(np.float64)
    rots = rotation6d(r6)
    return pos, rots


def geodesic_deg(r_pred: np.ndarray, r_gt: np.ndarray) -> float:
    rel = np.einsum("...ij,...ik->...jk", r_gt, r_pred)
    tr = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
    cos = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)).mean())


@torch.inference_mode()
def predict_episode_actions(
    wam: CosmosWAM,
    processed_branch: Path,
    dataset_name: str,
    episode_index: int,
    task_index: int,
    states_79: np.ndarray,
    gstats: dict,
    device: torch.device,
    batch_size: int = 32,
    ode_steps: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Run batched WAM v2.0 inference for all cached frames of an episode.

    Returns:
        frame_indices: [M] int64 parquet frame indices
        preds_47: [M, 10, 47] float32 denormalized predictions
    """
    dataset = processed_branch / "wam_lerobot_v2" / dataset_name
    latent_path = dataset / f"latents/chunk-000/episode_{episode_index:06d}.safetensors"
    with safe_open(str(latent_path), framework="pt", device="cpu") as src:
        frame_indices = src.get_tensor("frame_index").numpy().astype(np.int64)
        all_latents = src.get_tensor("latents")

    with safe_open(str(processed_branch / "prompt_bank_bf16.safetensors"), framework="pt", device="cpu") as bank:
        text_t0 = bank.get_tensor(f"task_{task_index}")

    st_mean = np.asarray(gstats["observation.state_5pt_hist"]["mean"], dtype=np.float32)
    st_std = np.maximum(np.asarray(gstats["observation.state_5pt_hist"]["std"], dtype=np.float32), 1e-6)
    act_mean = np.asarray(gstats["action"]["mean"], dtype=np.float32)
    act_std = np.maximum(np.asarray(gstats["action"]["std"], dtype=np.float32), 1e-6)

    n = len(frame_indices)
    preds = np.zeros((n, 10, 47), dtype=np.float32)
    text_buf = torch.empty((batch_size, *text_t0.shape), dtype=text_t0.dtype, device=device)
    for r in range(batch_size):
        text_buf[r].copy_(text_t0)

    gen = torch.Generator(device=device).manual_seed(20261123 + episode_index)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        bs = end - start
        f_idx = frame_indices[start:end]
        st_norm = (states_79[f_idx] - st_mean) / st_std
        proprio = torch.from_numpy(st_norm[:, :, :75]).to(device)
        ee_hist = torch.from_numpy(st_norm[:, :, 75:79]).to(device)
        video = all_latents[start:end].to(device)
        emb_id = torch.ones((bs,), dtype=torch.int64, device=device)
        noise = torch.randn((bs, 10, 47), device=device, generator=gen)
        with autocast(device):
            ctx = wam.encode_history(video, text_buf[:bs], proprio=proprio, ee_hist=ee_hist, embodiment_id=emb_id)
            out_norm = wam.sample(ctx, noise, ode_steps, embodiment_id=emb_id)
        preds[start:end] = out_norm.float().cpu().numpy() * act_std + act_mean
    return frame_indices, preds


def draw_3d_panel(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    cam,
    grid_lines,
    t0_pos: np.ndarray,
    t0_rots: np.ndarray,
    fut_pos: np.ndarray,
    fut_rots: np.ndarray,
    title: str,
    subtitle: str,
    accent: tuple[int, int, int],
    font_med,
    font_sm,
    show_legend: bool = False,
):
    x0, y0, w, h = rect
    draw.rectangle((x0, y0, x0 + w, y0 + h), fill=(12, 17, 26), outline=accent, width=2)
    draw.rectangle((x0 + 2, y0 + 2, x0 + w - 2, y0 + 46), fill=(19, 27, 40))
    draw.text((x0 + 10, y0 + 7), title, font=font_med, fill=accent)
    draw.text((x0 + 10, y0 + 26), subtitle, font=font_sm, fill=(155, 172, 196))

    # Ground grid
    for p_a, p_b in grid_lines:
        uv, _ = project_points(np.stack([p_a, p_b]), cam, rect, focal=500.0)
        draw.line([(uv[0, 0], uv[0, 1]), (uv[1, 0], uv[1, 1])], fill=(32, 44, 62), width=1)

    # Origin axes on ground
    orig = np.array([0.0, 0.0, -CANON_LEG_HEIGHT])
    for axis_vec, col, lbl in (
        (np.array([0.28, 0.0, 0.0]), (235, 85, 85), "+X"),
        (np.array([0.0, 0.28, 0.0]), (80, 215, 115), "+Y"),
        (np.array([0.0, 0.0, 0.28]), (85, 155, 255), "+Z"),
    ):
        uv, _ = project_points(np.stack([orig, orig + axis_vec]), cam, rect, focal=500.0)
        draw.line([(uv[0, 0], uv[0, 1]), (uv[1, 0], uv[1, 1])], fill=col, width=2)
        draw.text((uv[1, 0] + 3, uv[1, 1] - 6), lbl, font=font_sm, fill=col)

    # Future 1.0s (10-step) trajectories
    for p_idx in range(5):
        traj = np.concatenate([t0_pos[p_idx : p_idx + 1], fut_pos[:, p_idx]], axis=0)
        uv_traj, _ = project_points(traj, cam, rect, focal=500.0)
        col = POINT_COLORS[p_idx]
        dim_col = tuple(max(30, int(c * 0.78)) for c in col)
        pts_2d = [(float(u), float(v)) for u, v in uv_traj]
        draw.line(pts_2d, fill=dim_col, width=2)
        for step_i, (u, v) in enumerate(pts_2d[1:], start=1):
            r = 4.0 if step_i == len(pts_2d) - 1 else 2.5
            draw.ellipse((u - r, v - r, u + r, v + r), fill=col)

    # t0 skeleton links + final-step (t0 + 1.0s) dashed/dim skeleton
    uv_t0, _ = project_points(t0_pos, cam, rect, focal=500.0)
    uv_end, _ = project_points(fut_pos[-1], cam, rect, focal=500.0)
    for child in (1, 2, 3, 4):
        draw.line([(uv_t0[0, 0], uv_t0[0, 1]), (uv_t0[child, 0], uv_t0[child, 1])], fill=(175, 190, 210), width=2)
        draw.line([(uv_end[0, 0], uv_end[0, 1]), (uv_end[child, 0], uv_end[child, 1])], fill=(78, 92, 114), width=1)

    # Local RGB orientation triads at t0 and at final predicted/gt step
    for p_idx in range(5):
        p = t0_pos[p_idx]
        rot = t0_rots[p_idx]
        for ax_i, ax_col in enumerate(((240, 75, 75), (75, 220, 110), (80, 150, 255))):
            tip = p + rot[:, ax_i] * 0.10
            uv_ax, _ = project_points(np.stack([p, tip]), cam, rect, focal=500.0)
            draw.line([(uv_ax[0, 0], uv_ax[0, 1]), (uv_ax[1, 0], uv_ax[1, 1])], fill=ax_col, width=2)
        u, v = float(uv_t0[p_idx, 0]), float(uv_t0[p_idx, 1])
        r = 6 if p_idx == 0 else 5
        draw.ellipse((u - r, v - r, u + r, v + r), fill=POINT_COLORS[p_idx], outline=(255, 255, 255), width=1)

    if show_legend:
        for idx_l, (name, col) in enumerate(zip(POINT_NAMES, POINT_COLORS)):
            lx, ly = x0 + w - 88, y0 + 54 + idx_l * 18
            draw.ellipse((lx, ly + 3, lx + 8, ly + 11), fill=col)
            draw.text((lx + 14, ly), name, font=font_sm, fill=(215, 224, 238))


def render_episode(
    processed_branch: Path,
    dataset_name: str,
    title_prefix: str,
    episode_index: int,
    task_index: int,
    split: str,
    out_mp4: Path,
    wam: CosmosWAM,
    gstats: dict,
    device: torch.device,
    ckpt_step: int = 5000,
) -> dict:
    dataset = processed_branch / "wam_lerobot_v2" / dataset_name
    video_path = dataset / f"videos/chunk-000/observation.images.cam_head/episode_{episode_index:06d}.mp4"
    parquet_path = dataset / f"data/chunk-000/episode_{episode_index:06d}.parquet"

    frames_rgb = decode_video_rgb(video_path, 224, 224)
    cols = [
        "observation.state_5pt_hist",
        "action",
        "motion_bucket",
        "pelvis_disp_1s",
        "ankle_max_lift_1s",
        "wrist_disp_1s",
    ]
    tbl = pq.read_table(parquet_path, columns=cols)
    n_rows = tbl.num_rows
    states = (
        tbl["observation.state_5pt_hist"]
        .combine_chunks()
        .flatten()
        .flatten()
        .to_numpy(zero_copy_only=False)
        .astype(np.float32)
        .reshape(n_rows, 4, 79)
    )
    actions = (
        tbl["action"]
        .combine_chunks()
        .flatten()
        .flatten()
        .to_numpy(zero_copy_only=False)
        .astype(np.float32)
        .reshape(n_rows, 10, 47)
    )
    buckets = tbl["motion_bucket"].to_numpy()
    p_disp = tbl["pelvis_disp_1s"].to_numpy()
    a_lift = tbl["ankle_max_lift_1s"].to_numpy()
    w_disp = tbl["wrist_disp_1s"].to_numpy()

    frame_indices, preds_47 = predict_episode_actions(
        wam, processed_branch, dataset_name, episode_index, task_index, states, gstats, device
    )

    cam = make_camera(
        eye=np.array([-1.65, -1.40, 0.45], dtype=np.float64),
        target=np.array([0.20, 0.0, -0.34], dtype=np.float64),
    )
    rect_gt = (398, 60, 426, 568)
    rect_pred = (838, 60, 426, 568)

    font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 19)
    font_med = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
    font_body = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)

    grid_lines = []
    gz = -CANON_LEG_HEIGHT
    for x in np.linspace(-0.4, 1.0, 8):
        grid_lines.append((np.array([x, -0.6, gz]), np.array([x, 0.6, gz])))
    for y in np.linspace(-0.6, 0.6, 7):
        grid_lines.append((np.array([-0.4, y, gz]), np.array([1.0, y, gz])))

    proc = subprocess.Popen(
        [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1280x720", "-r", "10",
            "-i", "pipe:0",
            "-r", "25", "-c:v", "libx264", "-threads", "2", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(out_mp4),
        ],
        stdin=subprocess.PIPE,
    )

    ades, fdes, rots_err, low_ades, wri_ades = [], [], [], [], []
    try:
        for m_idx, f in enumerate(frame_indices):
            if f >= len(frames_rgb):
                break
            t0_feat45 = states[f, -1, :45]
            gt_feat45 = actions[f, :, :45]
            pr_feat45 = preds_47[m_idx, :, :45]

            t0_pos, t0_rots = canonical_45_to_meters_and_rots(t0_feat45)
            gt_pos, gt_rots = canonical_45_to_meters_and_rots(gt_feat45)
            pr_pos, pr_rots = canonical_45_to_meters_and_rots(pr_feat45)

            dist_mm = np.linalg.norm(pr_pos - gt_pos, axis=-1) * 1000.0  # [10, 5]
            ade = float(dist_mm.mean())
            fde = float(dist_mm[-1].mean())
            low_ade = float(dist_mm[:, :3].mean())
            wri_ade = float(dist_mm[:, 3:].mean())
            r_deg = geodesic_deg(pr_rots, gt_rots)
            ades.append(ade)
            fdes.append(fde)
            low_ades.append(low_ade)
            wri_ades.append(wri_ade)
            rots_err.append(r_deg)

            canvas = Image.new("RGB", (1280, 720), (16, 22, 33))
            draw = ImageDraw.Draw(canvas)

            # Top header
            draw.rectangle((0, 0, 1280, 50), fill=(22, 30, 44))
            draw.text(
                (16, 14),
                f"{title_prefix} task_{task_index} · ep {episode_index:06d} ({split}) · frame {f:03d}/{n_rows} ({f*0.1:.1f}s)   |   ADE {ade:.1f} mm  ·  Rot {r_deg:.1f}°",
                font=font_title,
                fill=(240, 245, 255),
            )
            b_label, b_color = BUCKET_META.get(int(buckets[f]), ("Bucket C · Idle", (110, 122, 138)))
            draw.rounded_rectangle((1060, 10, 1264, 40), radius=6, fill=b_color)
            draw.text((1076, 17), b_label, font=font_body, fill=(255, 255, 255))

            # Left panel: Ego-RGB (368x368) + Gripper comparison card below
            rgb_img = Image.fromarray(frames_rgb[f]).resize((368, 368), Image.Resampling.BICUBIC)
            canvas.paste(rgb_img, (16, 60))
            draw.rectangle((16, 60, 384, 428), outline=(58, 74, 98), width=2)
            draw.rectangle((16, 60, 232, 84), fill=(16, 22, 33))
            draw.text((22, 65), "Ego-RGB (cam_head · 10 Hz)", font=font_sm, fill=(205, 218, 236))

            # Metrics & Gripper card under RGB
            draw.rectangle((16, 442, 384, 628), fill=(19, 27, 40), outline=(58, 74, 98), width=1)
            draw.text((28, 452), "Frame Error (WAM v2.0 vs GT)", font=font_med, fill=(220, 232, 248))
            draw.text((28, 478), f"All 5-pt ADE:    {ade:5.1f} mm  (FDE {fde:5.1f} mm)", font=font_body, fill=(235, 242, 252))
            draw.text((28, 500), f"Lower-body ADE:  {low_ade:5.1f} mm", font=font_body, fill=(255, 200, 110))
            draw.text((28, 522), f"Wrists ADE:      {wri_ade:5.1f} mm", font=font_body, fill=(100, 225, 255))
            draw.text((28, 544), f"Mean Rot Error:  {r_deg:5.1f} deg", font=font_body, fill=(225, 165, 255))

            gt_gl, gt_gr = float(actions[f, 0, 45]), float(actions[f, 0, 46])
            pr_gl, pr_gr = float(preds_47[m_idx, 0, 45]), float(preds_47[m_idx, 0, 46])
            draw.text((28, 574), f"Gripper GT:   L={gt_gl:.2f}   R={gt_gr:.2f}", font=font_body, fill=(130, 225, 160))
            draw.text((28, 596), f"Gripper Pred: L={pr_gl:.2f}   R={pr_gr:.2f}", font=font_body, fill=(120, 195, 255))

            # Center panel: GT 5-point
            draw_3d_panel(
                draw, rect_gt, cam, grid_lines,
                t0_pos, t0_rots, gt_pos, gt_rots,
                title="LEFT: Ground Truth (5-pt + Future 1.0s)",
                subtitle="Canonical Human v2.0 {P_t0} Yaw-Only FLU (m)",
                accent=(95, 220, 145),
                font_med=font_med, font_sm=font_sm, show_legend=True,
            )

            # Right panel: WAM v2.0 Predicted 5-point
            draw_3d_panel(
                draw, rect_pred, cam, grid_lines,
                t0_pos, t0_rots, pr_pos, pr_rots,
                title="RIGHT: WAM v2.0 Prediction (5-pt + Future 1.0s)",
                subtitle=f"step-{ckpt_step} · 16-step ODE · emb_id=1",
                accent=(90, 185, 255),
                font_med=font_med, font_sm=font_sm, show_legend=False,
            )

            # Bottom telemetry bar
            draw.rectangle((0, 640, 1280, 720), fill=(22, 30, 44))
            draw.text(
                (16, 650),
                f"1s Motion GT: pelvis_disp={p_disp[f]:.3f}m   ankle_lift={a_lift[f]:.3f}m   wrist_disp={w_disp[f]:.3f}m   |   Running Mean: ADE={np.mean(ades):.1f}mm  FDE={np.mean(fdes):.1f}mm  Rot={np.mean(rots_err):.1f}°",
                font=font_body,
                fill=(235, 242, 252),
            )
            draw.text(
                (16, 680),
                f"t0 Canonical: Ankles Z/H_leg=({t0_feat45[5]:+.2f}, {t0_feat45[8]:+.2f})   L-Wrist=({t0_pos[3,0]:+.2f},{t0_pos[3,1]:+.2f},{t0_pos[3,2]:+.2f})m   R-Wrist=({t0_pos[4,0]:+.2f},{t0_pos[4,1]:+.2f},{t0_pos[4,2]:+.2f})m",
                font=font_sm,
                fill=(170, 186, 210),
            )

            proc.stdin.write(canvas.tobytes())
    finally:
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed for episode {episode_index}")

    return {
        "episode": episode_index,
        "split": split,
        "frames": len(ades),
        "ade_mm": round(float(np.mean(ades)), 2),
        "fde_mm": round(float(np.mean(fdes)), 2),
        "lower_ade_mm": round(float(np.mean(low_ades)), 2),
        "wrist_ade_mm": round(float(np.mean(wri_ades)), 2),
        "rot_deg": round(float(np.mean(rots_err)), 2),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p1-processed", type=Path, required=True)
    parser.add_argument("--wam-run", type=Path, required=True)
    parser.add_argument("--wam-checkpoint", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--backbone-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--branch", type=str, default="gripper")
    parser.add_argument("--dataset-name", type=str, default="g1_parallel_gripper")
    parser.add_argument("--title-prefix", type=str, default="OpenHLM")
    parser.add_argument("--scenario-key", type=str, default="openhlm-red-can-real-sample")
    parser.add_argument("--mode-key", type=str, default="real_sample")
    parser.add_argument("--video-prefix", type=str, default="openhlm-red-can-gt-vs-pred")
    parser.add_argument("--episodes", type=int, nargs="+", default=[8, 20, 32])
    args = parser.parse_args()

    device = torch.device("cuda:0")
    config = json.loads((args.wam_run / "run.json").read_text())
    saved = torch.load(args.wam_checkpoint, map_location="cpu", weights_only=True)
    dit_block_ids = {int(k.split(".")[2]) for k in saved["expert"] if k.startswith("backbone.transformer_blocks.")}
    wam = CosmosWAM(
        load_backbone(args.backbone, args.backbone_manifest, device),
        width=config["width"],
        layers=config["layers"],
        unfreeze_blocks=len(dit_block_ids),
        use_proprio=True,
        num_embodiments=config["num_embodiments"],
    ).to(device).eval()
    wam.load_state_dict(saved["expert"], strict=False)

    gstats = json.loads((args.p1_processed / "canonical_human_5pt_stats.json").read_text())
    branch_root = args.p1_processed / args.branch
    eps_meta = {
        int(row["episode_index"]): row
        for row in (
            json.loads(line)
            for line in (branch_root / "wam_lerobot_v2" / args.dataset_name / "meta/episodes.jsonl").read_text().splitlines()
            if line.strip()
        )
    }

    manifest_path = args.out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["cases"] = [c for c in manifest["cases"] if not (c.get("scenario") == args.scenario_key and c.get("mode") == args.mode_key)]

    summaries = []
    for ep in args.episodes:
        row = eps_meta[ep]
        split = row["split"]
        task_index = int(row["task_index"])
        dst_name = f"{args.video_prefix}-ep{ep:06d}-{split}.mp4"
        dst_path = args.out_dir / dst_name
        print(f"Inferring & rendering {dst_name} (task_{task_index})...", flush=True)
        metrics = render_episode(
            branch_root, args.dataset_name, args.title_prefix, ep, task_index, split,
            dst_path, wam, gstats, device, ckpt_step=int(saved["step"]),
        )
        summaries.append(metrics)
        print(json.dumps(metrics), flush=True)
        manifest["cases"].append({
            "video": dst_name,
            "scenario": args.scenario_key,
            "seed": ep,
            "mode": args.mode_key,
            "robot": "g1_real",
            "bridge": "canonical_human_v2_gt_vs_pred",
            "success": True,
            "termination": f"ADE {metrics['ade_mm']:.1f}mm / Rot {metrics['rot_deg']:.1f}°",
            "steps": metrics["frames"] * 5,
            "min_wrist_distance_m": metrics["wrist_ade_mm"] / 1000.0,
            "hand_contact_steps": 0,
            "button_max_travel_m": None,
            "planning_calls": [],
        })

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"Done. Total cases in manifest: {len(manifest['cases'])}", flush=True)


if __name__ == "__main__":
    main()
