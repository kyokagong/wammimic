"""Diagnose why WAM v2.0 does or does not predict forward walking in MuJoCo vs real OpenHLM."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import mujoco
import numpy as np
import pyarrow.parquet as pq
from safetensors import safe_open
import torch

from morphmimic.env.env_cfg import COMMAND_NAME, morphmimic_env_cfg
from morphmimic.model.state import split_state
from preview.cpu_backend import CpuPreviewEnv
from scripts.evaluate_geniesim_indist import SCALE_Q1, build_red_can_scene, prepare_reference
from scripts.evaluate_geniesim_wam_mimic import DT, SLOTS, predict
from wammimic.data_pipeline.cache import VideoEncoder
from wammimic.data_pipeline.canonical_human import (
    CANON_LEG_HEIGHT,
    ROBOT_GEOMETRY,
    world_flu_to_canonical_history,
)
from wammimic.training.model import CosmosWAM, load_backbone


def decode_real_frames(mp4_path: Path, frame_ids: list[int]) -> list[np.ndarray]:
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", str(mp4_path),
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "224x224", "pipe:1",
    ]
    raw = subprocess.check_output(cmd)
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 224, 224, 3)
    return [arr[i].copy() for i in frame_ids]


def summarize_block(name: str, pred_47: np.ndarray):
    pos = pred_47[:, :15].reshape(10, 5, 3).copy()
    pos[:, :3, :] *= CANON_LEG_HEIGHT  # canonical human meters
    pel_x = pos[:, 0, 0]
    la_x, ra_x = pos[:, 1, 0], pos[:, 2, 0]
    la_z, ra_z = pos[:, 1, 2], pos[:, 2, 2]
    lr_dx = la_x - ra_x
    lr_dz = la_z - ra_z
    print(f"\n=== {name} ===")
    print(f"  Pelvis X (0.1s..1.0s, m): {[round(float(x), 3) for x in pel_x]}")
    print(f"  L-Ankle X (m):            {[round(float(x), 3) for x in la_x]}")
    print(f"  R-Ankle X (m):            {[round(float(x), 3) for x in ra_x]}")
    print(f"  L-Ankle Z (m, rest=-.85): {[round(float(z), 3) for z in la_z]}")
    print(f"  R-Ankle Z (m, rest=-.85): {[round(float(z), 3) for z in ra_z]}")
    print(f"  Summary: pel_dx_1s={pel_x[-1]:+.3f}m | max_ankle_lift={max(la_z.max(), ra_z.max()) + 0.85:+.3f}m | max|LA_x-RA_x|={np.abs(lr_dx).max():.3f}m | max|LA_z-RA_z|={np.abs(lr_dz).max():.3f}m")


def main():
    device = "cuda:0"
    run = Path("/data/kk/wammimic/runs/geniesim-wam-mimic-20261120")
    wam_run = Path("/data/kk/wammimic/runs/wam-v2-canonical-20k-2gpu-b64-20261123")
    wam_ckpt = wam_run / "checkpoints/step-00020000.pt"
    mimic_run = Path("/data/kk/wammimic/runs/mimic-all-motion-resume12000-40000-20261122")
    p1_proc = Path("/data/kk/wammimic/runs/p1-data-20261120/processed")

    config = json.loads((wam_run / "run.json").read_text())
    saved = torch.load(wam_ckpt, map_location="cpu", weights_only=True)
    dit_blocks = {int(k.split(".")[2]) for k in saved["expert"] if k.startswith("backbone.transformer_blocks.")}
    wam = CosmosWAM(
        load_backbone(Path("/data/kk/models/nvidia/Cosmos-Predict2.5-2B"),
                      Path("/data/kk/data/wammimic_downloads_20261116/cosmos_backbone_20261118.json"), device),
        width=config["width"], layers=config["layers"], unfreeze_blocks=len(dit_blocks),
        use_proprio=True, num_embodiments=config["num_embodiments"],
    ).to(device).eval()
    wam.load_state_dict(saved["expert"], strict=False)
    vae = VideoEncoder("/data/kk/models/wammimic_cosmos25_encoders_20261118")

    gstats = json.loads((p1_proc / "canonical_human_5pt_stats.json").read_text())
    act_mean = np.asarray(gstats["action"]["mean"], dtype=np.float32)
    act_std = np.maximum(np.asarray(gstats["action"]["std"], dtype=np.float32), 1e-6)
    prop_mean = np.asarray(gstats["proprio_5pt"]["mean"], dtype=np.float32)
    prop_std = np.maximum(np.asarray(gstats["proprio_5pt"]["std"], dtype=np.float32), 1e-6)
    st_mean = np.asarray(gstats["observation.state_5pt_hist"]["mean"], dtype=np.float32)
    st_std = np.maximum(np.asarray(gstats["observation.state_5pt_hist"]["std"], dtype=np.float32), 1e-6)

    with safe_open(str(p1_proc / "gripper/prompt_bank_bf16.safetensors"), framework="pt", device="cpu") as f:
        text_t0 = f.get_tensor("task_0")[None].to(device)

    # 1. Capture MuJoCo 3.2m stationary RGB + Q1 proprio
    mcfg = json.loads((mimic_run / "config.json").read_text())
    mdata = Path(mcfg["prepared_manifest"]).parent
    subset_yaml, rests = prepare_reference(run / "diag_ref", mdata, "q1", np.array([-3.2, 0.0, 0.0]), 60)
    cfg = morphmimic_env_cfg("q1", subset_yaml=str(subset_yaml), num_envs=1, play=True, eval=True,
                             goal_anchor=mcfg["goal_anchor"], ref_feedback=True)
    cfg.scene.spec_fn = build_red_can_scene(run / "assets", scale=SCALE_Q1, robot="q1", fovy=72.0)
    env = CpuPreviewEnv(cfg, device="cpu")
    try:
        state_ids = torch.arange(1, device=env.device)
        env.command_manager.get_term(COMMAND_NAME).assign_clips(state_ids, torch.zeros_like(state_ids))
        env.reset(env_ids=state_ids)
        parts = split_state(env.state())
        actual_pos = parts.body_pos[0, list(SLOTS)].cpu().numpy()
        actual_quat = parts.body_quat[0, list(SLOTS)].cpu().numpy()
        p5 = np.concatenate((actual_pos, actual_quat), axis=-1)
        poses_50 = np.repeat(p5[None], 20, axis=0)
        times_50 = np.arange(20, dtype=np.float64) * DT
        geom = ROBOT_GEOMETRY["q1"]
        q1_prop_raw = world_flu_to_canonical_history(
            poses_50, times_50, np.array([[0, 5, 10, 15]]), geom["leg_height"], geom["arm_length"], allow_zero_initial=True
        )[0]
        q1_prop_norm = ((q1_prop_raw - prop_mean) / prop_std).astype(np.float32)

        head = mujoco.Renderer(env.sim.mj_model, height=224, width=224)
        head.update_scene(env.native_data, camera="wam_head")
        mj_img = head.render().copy()
        head.close()
        mj_frames = [mj_img] * 4
    finally:
        env.close()

    # 2. Load real OpenHLM ep 000032 (test) at frame 10 (just before walking) and frame 43 (mid-walk)
    ds = p1_proc / "gripper/wam_lerobot_v2/g1_parallel_gripper"
    tbl = pq.read_table(ds / "data/chunk-000/episode_000032.parquet", columns=["observation.state_5pt_hist", "action"])
    real_states = tbl["observation.state_5pt_hist"].combine_chunks().flatten().flatten().to_numpy(zero_copy_only=False).astype(np.float32).reshape(-1, 4, 79)
    real_actions = tbl["action"].combine_chunks().flatten().flatten().to_numpy(zero_copy_only=False).astype(np.float32).reshape(-1, 10, 47)

    mp4 = ds / "videos/chunk-000/observation.images.cam_head/episode_000032.mp4"
    real_frames_f10 = decode_real_frames(mp4, [7, 8, 9, 10])
    real_frames_f43 = decode_real_frames(mp4, [40, 41, 42, 43])

    real_st_f10 = ((real_states[10] - st_mean) / st_std).astype(np.float32)
    real_st_f43 = ((real_states[43] - st_mean) / st_std).astype(np.float32)
    ee_zero = np.zeros((4, 4), dtype=np.float32)

    print("Q1 stationary proprio_raw t0 ankles Z/H_leg:", q1_prop_raw[-1, 5], q1_prop_raw[-1, 8], "wrists:", np.round(q1_prop_raw[-1, 9:15], 3))
    print("Real ep32 f10 proprio_raw t0 ankles Z/H_leg:", real_states[10, -1, 5], real_states[10, -1, 8], "wrists:", np.round(real_states[10, -1, 9:15], 3))

    def run_pred(name, frames, prop, ee, emb):
        gen = torch.Generator(device=device).manual_seed(0)
        out = predict(wam, vae, frames, text_t0, gen, act_mean, act_std, proprio=prop, ee_hist=ee, embodiment_id=emb)
        summarize_block(name, out)

    run_pred("Cond A: MuJoCo 3.2m RGB + Q1 stationary proprio (emb_id=1)", mj_frames, q1_prop_norm, ee_zero, 1)
    run_pred("Cond B: MuJoCo 3.2m RGB + Q1 stationary proprio (emb_id=2 human)", mj_frames, q1_prop_norm, ee_zero, 2)
    run_pred("Cond C: REAL OpenHLM f10 RGB + Q1 stationary proprio (emb_id=1)", real_frames_f10, q1_prop_norm, ee_zero, 1)
    run_pred("Cond D: REAL OpenHLM f10 RGB + REAL f10 proprio (emb_id=1)", real_frames_f10, real_st_f10[:, :75], real_st_f10[:, 75:79], 1)
    summarize_block("GT Reference: REAL OpenHLM ep32 f10 ground truth", real_actions[10])
    run_pred("Cond E: MuJoCo 3.2m RGB + REAL f43 walking proprio (emb_id=1)", mj_frames, real_st_f43[:, :75], real_st_f43[:, 75:79], 1)
    run_pred("Cond F: REAL OpenHLM f43 RGB + REAL f43 walking proprio (emb_id=1)", real_frames_f43, real_st_f43[:, :75], real_st_f43[:, 75:79], 1)


if __name__ == "__main__":
    main()
