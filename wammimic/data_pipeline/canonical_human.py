"""Canonical Human (H_5') five-point geometry, motion bucketing, global stats, and 6-gate QA (v2.0).

Implements docs/design/data-processing.md:
- Unified ROS REP-103 FLU right-handed Z-up frame (+X forward, +Y left, +Z up)
- Point order: [0: Pelvis, 1: LeftAnkle, 2: RightAnkle, 3: LeftWrist, 4: RightWrist]
- Anchor frame {P_t0}: pelvis world position at t0, Yaw-Only heading H_0 = R_z(psi_t0)
- Dual-scale Canonical Human retargeting:
    * Lower body (0..2): divided by measured standing pelvis-ankle height H_leg^(s)
      (so standing ankles sit at z ≈ -1.00)
    * Wrists (3..4): pelvis translation scaled by H_leg_canon / H_leg^(s)
      + relative-to-pelvis arm vector scaled by L_arm_canon / L_arm^(s)
- Motion diversity tags (pelvis_disp_1s, ankle_max_lift_1s, wrist_disp_1s) & buckets (A/B/C)
- Single global canonical_human_5pt_stats.json fitted on active train splits
- Six mandatory QA gates before training ingestion
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

from .common import sha256, write_json
from .geometry import matrices, rotation6d

CANONICAL_SCHEMA = "canonical-human-5pt-v2"
CANON_LEG_HEIGHT = 0.85  # meters (standard adult standing pelvis-to-ankle vertical height)
CANON_ARM_LENGTH = 0.58  # meters (standard adult shoulder-to-wrist effective arm length)

# Measured robot geometry constants (URDF / standing FK)
ROBOT_GEOMETRY = {
    "g1": {"leg_height": 0.745, "arm_length": 0.50, "ankle_rest_z": 0.050},
    "q1": {"leg_height": 0.379, "arm_length": 0.36, "ankle_rest_z": 0.045},
    "bumi": {"leg_height": 0.440, "arm_length": 0.39, "ankle_rest_z": 0.045},
}

# Right-multiplication matrix converting NOKOV local bone axes (X=Left, Y=Up, Z=Fwd)
# to ROS REP-103 FLU (X=Fwd, Y=Left, Z=Up): R_flu = R_nokov @ C_NOKOV_TO_FLU.
# det(C) = +1 (proper SO(3) rotation).
C_NOKOV_TO_FLU = np.array(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)

EMBODIMENT_IDS = {
    "bimanual_stub": 0,
    "bimanual_parallel_gripper": 1,
    "human_5pt_only": 2,
}

BUCKET_LOCOMOTION = 0  # Bucket A: pelvis_disp_1s >= 0.15m or ankle_max_lift_1s >= 0.06m
BUCKET_MANIPULATION = 1  # Bucket B: not A and wrist_disp_1s >= 0.12m
BUCKET_IDLE = 2  # Bucket C: remaining low-speed / static windows
BUCKET_NAMES = {
    BUCKET_LOCOMOTION: "locomotion",
    BUCKET_MANIPULATION: "manipulation",
    BUCKET_IDLE: "idle",
}


def _normalize_c_rest(c_rest: np.ndarray | None) -> np.ndarray | None:
    if c_rest is None:
        return None
    c = np.asarray(c_rest, dtype=np.float64)
    if c.shape == (3, 3):
        return np.broadcast_to(c, (5, 3, 3)).copy()
    if c.shape == (5, 3, 3):
        return c.copy()
    raise ValueError(f"Expected C_rest of shape (3,3) or (5,3,3), got {c.shape}")


def heading_matrix_from_pelvis_rot(pelvis_rot: np.ndarray) -> np.ndarray:
    """Compute Yaw-Only heading matrix H_0 = R_z(psi) from pelvis 3x3 FLU rotation(s).

    Accepts shape (..., 3, 3) and returns shape (..., 3, 3) in float64.
    """
    r = np.asarray(pelvis_rot, dtype=np.float64)
    yaw = np.arctan2(r[..., 1, 0], r[..., 0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    h = np.zeros(r.shape, dtype=np.float64)
    h[..., 0, 0] = c
    h[..., 0, 1] = -s
    h[..., 1, 0] = s
    h[..., 1, 1] = c
    h[..., 2, 2] = 1.0
    return h


def calibrate_world_rotations(quaternions_wxyz: np.ndarray, c_rest: np.ndarray | None = None) -> np.ndarray:
    """Convert [..., 5, 4] wxyz quaternions to [..., 5, 3, 3] FLU rotation matrices."""
    rots = matrices(quaternions_wxyz)
    c = _normalize_c_rest(c_rest)
    if c is not None:
        rots = np.einsum("...pij,pjk->...pik", rots, c)
    return rots


def measure_standing_leg_height(poses_world: np.ndarray, valid: np.ndarray | None = None) -> float:
    """Measure standing pelvis-to-ankle vertical distance H_leg^(s) from world Z-up poses [N, 5, 7].

    Uses the 85th percentile of (z_pelvis - 0.5*(z_LA + z_RA)) over valid frames so brief
    squats/bending do not bias standing height downward.
    """
    p = poses_world[valid] if valid is not None else poses_world
    if len(p) == 0:
        raise ValueError("No valid poses to measure standing leg height")
    dz = p[:, 0, 2] - 0.5 * (p[:, 1, 2] + p[:, 2, 2])
    dz = dz[np.isfinite(dz) & (dz > 0.25) & (dz < 1.40)]
    if len(dz) == 0:
        raise ValueError("Standing pelvis-ankle height out of physical bounds")
    return float(np.percentile(dz, 85))


def world_flu_to_canonical_5pt(
    future_world: np.ndarray,
    anchor_world: np.ndarray,
    leg_height: float,
    arm_length: float,
    *,
    c_rest: np.ndarray | None = None,
) -> np.ndarray:
    """Project world Z-up FLU 5-point poses into Canonical Human Yaw-Only {P_t0} 45D features.

    Args:
        future_world: [..., T, 5, 7] or [..., 5, 7] world poses (xyz + wxyz).
        anchor_world: [..., 5, 7] world pose at t0 (broadcastable to future_world without T).
        leg_height: measured standing pelvis-ankle vertical height H_leg^(s) in meters.
        arm_length: measured effective shoulder-wrist arm length L_arm^(s) in meters.
        c_rest: optional (3,3) or (5,3,3) local bone axis calibration matrix to FLU.

    Returns:
        [..., T, 45] (or [..., 45]) float32: 15D positions + 30D rot6d.
    """
    if not (0.25 < leg_height < 1.50):
        raise ValueError(f"Invalid leg_height: {leg_height}")
    if not (0.20 < arm_length < 1.10):
        raise ValueError(f"Invalid arm_length: {arm_length}")

    future = np.asarray(future_world, dtype=np.float64)
    anchor = np.asarray(anchor_world, dtype=np.float64)
    if future.shape[-2:] != (5, 7) or anchor.shape[-2:] != (5, 7):
        raise ValueError(f"Expected (...,5,7) poses, got future={future.shape}, anchor={anchor.shape}")
    if not (np.isfinite(future).all() and np.isfinite(anchor).all()):
        raise ValueError("Non-finite pose in world_flu_to_canonical_5pt")

    single_step = future.ndim == anchor.ndim
    if single_step:
        future = future[..., None, :, :]

    # Yaw-Only heading H_0 from calibrated t0 pelvis orientation
    anchor_rots = calibrate_world_rotations(anchor[..., :, 3:], c_rest)
    h0 = heading_matrix_from_pelvis_rot(anchor_rots[..., 0, :, :])  # [..., 3, 3]
    p0_t0 = anchor[..., 0, :3]  # [..., 3]

    # World positions relative to t0 pelvis, rotated into {P_t0}
    pos_rel_t0 = future[..., :, :3] - p0_t0[..., None, None, :]  # [..., T, 5, 3]
    pos_h0 = np.einsum("...tpj,...jk->...tpk", pos_rel_t0, h0)

    out_pos = np.empty_like(pos_h0)
    # Lower body (pelvis, LA, RA): divided by H_leg^(s) so standing ankles ≈ -1.00
    out_pos[..., :3, :] = pos_h0[..., :3, :] / leg_height

    # Wrists (LW, RW): pelvis translation scaled by H_leg_canon/H_leg^(s)
    #                  + relative wrist-to-pelvis scaled by L_arm_canon/L_arm^(s)
    pelvis_shift_h0 = pos_h0[..., 0:1, :]  # [..., T, 1, 3]
    wrist_rel_pelvis_h0 = pos_h0[..., 3:5, :] - pelvis_shift_h0  # [..., T, 2, 3]
    out_pos[..., 3:5, :] = (
        pelvis_shift_h0 * (CANON_LEG_HEIGHT / leg_height)
        + wrist_rel_pelvis_h0 * (CANON_ARM_LENGTH / arm_length)
    )

    # Rotations relative to Yaw-Only H_0
    future_rots = calibrate_world_rotations(future[..., :, 3:], c_rest)  # [..., T, 5, 3, 3]
    rel_rots = np.einsum("...ij,...tpjk->...tpik", h0.swapaxes(-1, -2), future_rots)
    rot6d = rel_rots[..., :, :2].swapaxes(-1, -2).reshape(future.shape[:-2] + (30,))

    feat = np.concatenate([out_pos.reshape(future.shape[:-2] + (15,)), rot6d], axis=-1).astype(np.float32)
    return feat[..., 0, :] if single_step else feat


def canonical_5pt_to_target_world(
    block_45: np.ndarray,
    anchor_world: np.ndarray,
    target_leg_height: float,
    target_arm_length: float,
    *,
    ankle_min_z: float | None = None,
    c_rest: np.ndarray | None = None,
) -> np.ndarray:
    """Inverse bridge: Canonical Human 45D (Yaw-Only {P_t0}) -> target robot/human world FLU [..., 5, 7].

    Exact inverse of world_flu_to_canonical_5pt when target geometry == source geometry
    and ankle_min_z is None.
    """
    block = np.asarray(block_45, dtype=np.float64)
    anchor = np.asarray(anchor_world, dtype=np.float64)
    if block.shape[-1] != 45 or anchor.shape[-2:] != (5, 7):
        raise ValueError(f"Unexpected shapes: block={block.shape}, anchor={anchor.shape}")

    single_step = block.ndim == anchor.ndim - 1
    if single_step:
        block = block[..., None, :]

    anchor_rots = calibrate_world_rotations(anchor[..., :, 3:], c_rest)
    h0 = heading_matrix_from_pelvis_rot(anchor_rots[..., 0, :, :])  # [..., 3, 3]
    p0_t0 = anchor[..., 0, :3]  # [..., 3]

    canon_pos = block[..., :15].reshape(block.shape[:-1] + (5, 3))
    canon_r6 = block[..., 15:].reshape(block.shape[:-1] + (5, 6))

    target_pos_h0 = np.empty_like(canon_pos)
    # Lower body: multiply by target H_leg
    target_pos_h0[..., :3, :] = canon_pos[..., :3, :] * target_leg_height
    # Wrists: extract canonical pelvis shift & relative wrist offset, rescale to target
    canon_pelvis_shift_m = canon_pos[..., 0:1, :] * CANON_LEG_HEIGHT
    canon_wrist_rel_m = canon_pos[..., 3:5, :] - canon_pelvis_shift_m
    target_pos_h0[..., 3:5, :] = (
        target_pos_h0[..., 0:1, :]
        + canon_wrist_rel_m * (target_arm_length / CANON_ARM_LENGTH)
    )

    world_pos = p0_t0[..., None, None, :] + np.einsum("...ij,...tpj->...tpi", h0, target_pos_h0)
    if ankle_min_z is not None:
        world_pos[..., 1:3, 2] = np.maximum(world_pos[..., 1:3, 2], float(ankle_min_z))

    rel_rots = rotation6d(canon_r6)  # [..., T, 5, 3, 3]
    world_rots = np.einsum("...ij,...tpjk->...tpik", h0, rel_rots)
    c = _normalize_c_rest(c_rest)
    if c is not None:
        world_rots = np.einsum("...tpij,pkj->...tpik", world_rots, c)  # right-multiply by C^T

    q_xyzw = Rotation.from_matrix(world_rots.reshape(-1, 3, 3)).as_quat()
    q_wxyz = q_xyzw[:, [3, 0, 1, 2]].reshape(block.shape[:-1] + (5, 4))
    out = np.concatenate([world_pos, q_wxyz], axis=-1).astype(np.float32)
    return out[..., 0, :, :] if single_step else out


def world_flu_to_canonical_history(
    poses_world: np.ndarray,
    times: np.ndarray,
    indices: np.ndarray,
    leg_height: float,
    arm_length: float,
    *,
    c_rest: np.ndarray | None = None,
    allow_zero_initial: bool = False,
) -> np.ndarray:
    """Compute strictly causal 4-frame Canonical Human proprio history [B, 4, 75].

    Each frame's linear/angular velocity uses ONLY backward difference (indices vs indices - 1).
    When allow_zero_initial=True, an index == 0 uses zero velocity (no future leakage).
    All 4 frames are expressed in the t0 (indices[:, -1]) Yaw-Only {P_t0} frame.
    """
    idx = np.asarray(indices, dtype=np.int64)
    if idx.ndim != 2 or idx.shape[1] != 4:
        raise ValueError("History requires shape [B, 4] indices")
    if not allow_zero_initial and (idx <= 0).any():
        raise ValueError("History requires shape [B, 4] indices with measured predecessors (> 0)")
    if (idx < 0).any() or (idx >= len(poses_world)).any() or (np.diff(idx, axis=1) <= 0).any():
        raise ValueError("History indices must be strictly increasing and in range")

    prev_idx = np.maximum(idx - 1, 0)
    is_zero = idx == 0
    current = np.asarray(poses_world[idx], dtype=np.float64)
    previous = np.asarray(poses_world[prev_idx], dtype=np.float64)
    dt = np.asarray(times[idx] - times[prev_idx], dtype=np.float64)
    if not np.isfinite(dt).all() or (dt[~is_zero] <= 0).any():
        raise ValueError("History requires strictly increasing finite timestamps")
    safe_dt = np.where(is_zero, 1.0, dt)

    anchor = current[:, -1]  # [B, 5, 7] at t0
    anchor_rots = calibrate_world_rotations(anchor[:, :, 3:], c_rest)
    h0 = heading_matrix_from_pelvis_rot(anchor_rots[:, 0, :, :])  # [B, 3, 3]

    # World linear velocity [B, 4, 5, 3] -> project into H_0 and dual-scale
    v_world = (current[..., :3] - previous[..., :3]) / safe_dt[..., None, None]
    if is_zero.any():
        v_world[is_zero] = 0.0
    v_h0 = np.einsum("btpj,bjk->btpk", v_world, h0)
    lin = np.empty_like(v_h0)
    lin[..., :3, :] = v_h0[..., :3, :] / leg_height
    v_pelvis_h0 = v_h0[..., 0:1, :]
    lin[..., 3:5, :] = (
        v_pelvis_h0 * (CANON_LEG_HEIGHT / leg_height)
        + (v_h0[..., 3:5, :] - v_pelvis_h0) * (CANON_ARM_LENGTH / arm_length)
    )

    # Spatial angular velocity in world FLU: invariant to constant right-multiplication C_rest
    r_curr = matrices(current[..., 3:])
    r_prev = matrices(previous[..., 3:])
    delta_r = r_curr @ r_prev.swapaxes(-1, -2)
    omega_world = Rotation.from_matrix(delta_r.reshape(-1, 3, 3)).as_rotvec().reshape(*idx.shape, 5, 3) / safe_dt[..., None, None]
    if is_zero.any():
        omega_world[is_zero] = 0.0
    ang = np.einsum("btpj,bjk->btpk", omega_world, h0)

    pose_45 = world_flu_to_canonical_5pt(current, anchor, leg_height, arm_length, c_rest=c_rest)
    result = np.concatenate(
        [pose_45, lin.reshape(*idx.shape, 15).astype(np.float32), ang.reshape(*idx.shape, 15).astype(np.float32)],
        axis=-1,
    )
    if not np.isfinite(result).all():
        raise ValueError("Non-finite canonical history")
    return result


def reproject_old_history_to_canonical(
    poses_10hz_world: np.ndarray,
    old_state_75: np.ndarray,
    valid_frames: np.ndarray,
    old_leg_length: float,
    leg_height: float,
    arm_length: float,
    *,
    c_rest: np.ndarray | None = None,
) -> np.ndarray:
    """Reproject existing 120Hz-velocity old_state_75 (full 3D pelvis frame) to Canonical Human Yaw-Only [N, 4, 75].

    Used when raw 120Hz mocap files are archived and only 10Hz poses_world + old parquet remain.
    Exact algebraic recovery: v_world = R_pelvis_old(t0) @ v_old, then project to H_0 with dual-scale.
    """
    n = len(poses_10hz_world)
    out = np.zeros((n, 4, 75), dtype=np.float32)
    if len(valid_frames) == 0:
        return out

    hist_idx = valid_frames[:, None] + np.arange(-3, 1, dtype=np.int64)
    current = np.asarray(poses_10hz_world[hist_idx], dtype=np.float64)  # [B, 4, 5, 7]
    anchor = current[:, -1]  # [B, 5, 7]

    # Exact 45D poses in Canonical Human Yaw-Only frame
    pose_45 = world_flu_to_canonical_5pt(current, anchor, leg_height, arm_length, c_rest=c_rest)

    # Recover world velocities from old full-3D pelvis frame
    r_pelvis_old = matrices(anchor[:, 0, 3:])  # [B, 3, 3]
    old_lin = np.asarray(old_state_75[valid_frames, :, 45:60], dtype=np.float64).reshape(-1, 4, 5, 3)
    old_lin[..., :3, :] *= float(old_leg_length)
    v_world = np.einsum("bij,btpj->btpi", r_pelvis_old, old_lin)

    old_ang = np.asarray(old_state_75[valid_frames, :, 60:75], dtype=np.float64).reshape(-1, 4, 5, 3)
    omega_world = np.einsum("bij,btpj->btpi", r_pelvis_old, old_ang)

    # Project into Yaw-Only H_0 and apply dual-scale
    anchor_rots = calibrate_world_rotations(anchor[:, :, 3:], c_rest)
    h0 = heading_matrix_from_pelvis_rot(anchor_rots[:, 0, :, :])  # [B, 3, 3]
    v_h0 = np.einsum("btpj,bjk->btpk", v_world, h0)
    lin = np.empty_like(v_h0)
    lin[..., :3, :] = v_h0[..., :3, :] / leg_height
    v_pelvis_h0 = v_h0[..., 0:1, :]
    lin[..., 3:5, :] = (
        v_pelvis_h0 * (CANON_LEG_HEIGHT / leg_height)
        + (v_h0[..., 3:5, :] - v_pelvis_h0) * (CANON_ARM_LENGTH / arm_length)
    )
    ang = np.einsum("btpj,bjk->btpk", omega_world, h0)

    out[valid_frames] = np.concatenate(
        [pose_45, lin.reshape(-1, 4, 15).astype(np.float32), ang.reshape(-1, 4, 15).astype(np.float32)],
        axis=-1,
    )
    if not np.isfinite(out[valid_frames]).all():
        raise ValueError("Non-finite reprojected canonical history")
    return out


def compute_motion_metrics(
    action_45: np.ndarray,
    state_t0_45: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute Canonical Human motion diversity tags and bucket IDs for N windows.

    Args:
        action_45: [N, 10, 45] canonical 5pt future actions (t0+0.1s .. t0+1.0s).
        state_t0_45: [N, 45] canonical 5pt pose at t0 (last frame of proprio history).

    Returns:
        (pelvis_disp_1s, ankle_max_lift_1s, wrist_disp_1s, motion_bucket)
        all length-N arrays (float32, float32, float32, int8).
    """
    act_pos = np.asarray(action_45[..., :15], dtype=np.float64).reshape(-1, 10, 5, 3)
    t0_pos = np.asarray(state_t0_45[..., :15], dtype=np.float64).reshape(-1, 5, 3)

    # 1. Pelvis 1.0s horizontal displacement in canonical meters
    pelvis_xy_end_m = act_pos[:, -1, 0, :2] * CANON_LEG_HEIGHT
    pelvis_disp_1s = np.linalg.norm(pelvis_xy_end_m, axis=-1).astype(np.float32)

    # 2. Max ankle lift over 1.0s relative to t0 ankle height (in canonical meters)
    dz_la = (act_pos[:, :, 1, 2] - t0_pos[:, None, 1, 2]) * CANON_LEG_HEIGHT
    dz_ra = (act_pos[:, :, 2, 2] - t0_pos[:, None, 2, 2]) * CANON_LEG_HEIGHT
    ankle_max_lift_1s = np.maximum(0.0, np.maximum(dz_la.max(axis=1), dz_ra.max(axis=1))).astype(np.float32)

    # 3. Max wrist displacement relative to pelvis over 1.0s (already in canonical meters)
    pelvis_shift_m = act_pos[:, :, 0:1, :] * CANON_LEG_HEIGHT
    wrist_rel_act = act_pos[:, :, 3:5, :] - pelvis_shift_m  # [N, 10, 2, 3]
    wrist_rel_t0 = t0_pos[:, None, 3:5, :]  # pelvis at t0 is [0,0,0]
    w_disp = np.linalg.norm(wrist_rel_act - wrist_rel_t0, axis=-1)  # [N, 10, 2]
    wrist_disp_1s = w_disp.max(axis=(1, 2)).astype(np.float32)

    # Bucket assignment: A (locomotion) -> B (manipulation) -> C (idle)
    bucket = np.full(len(pelvis_disp_1s), BUCKET_IDLE, dtype=np.int8)
    is_a = (pelvis_disp_1s >= 0.15) | (ankle_max_lift_1s >= 0.06)
    is_b = (~is_a) & (wrist_disp_1s >= 0.12)
    bucket[is_b] = BUCKET_MANIPULATION
    bucket[is_a] = BUCKET_LOCOMOTION
    return pelvis_disp_1s, ankle_max_lift_1s, wrist_disp_1s, bucket


def _streaming_moments_update(
    count: int,
    mean: np.ndarray,
    m2: np.ndarray,
    batch: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Parallel Welford merge for numerically stable mean/variance over millions of rows."""
    b_n = len(batch)
    if b_n == 0:
        return count, mean, m2
    b_mean = batch.mean(axis=0, dtype=np.float64)
    b_m2 = ((batch.astype(np.float64) - b_mean) ** 2).sum(axis=0)
    if count == 0:
        return b_n, b_mean, b_m2
    total = count + b_n
    delta = b_mean - mean
    new_mean = mean + delta * (b_n / total)
    new_m2 = m2 + b_m2 + (delta ** 2) * (count * b_n / total)
    return total, new_mean, new_m2


def fit_global_canonical_stats(
    branch_specs: list[dict[str, Any]],
    output_path: Path,
) -> dict[str, Any]:
    """Fit single global canonical_human_5pt_stats.json over active train splits.

    Each item in branch_specs:
      {
        "name": str,
        "dataset_dir": Path,           # e.g. .../wam_lerobot_v2/human_ego
        "train_episodes": list[int],
        "state_col": str,              # "observation.state" or "observation.state_5pt_hist"
        "weight": float,               # branch target sampling weight (e.g. 0.25)
        "has_ee": bool,
      }
    Computes weighted global mean/std across branches (each branch normalized to its
    target sampling weight) so no single huge branch dominates the shared 5pt statistics.
    """
    per_branch_stats: dict[str, Any] = {}
    ee_act_cnt, ee_act_mean, ee_act_m2 = 0, np.zeros(2), np.zeros(2)
    ee_hist_cnt, ee_hist_mean, ee_hist_m2 = 0, np.zeros(4), np.zeros(4)

    for spec in branch_specs:
        ds_dir = Path(spec["dataset_dir"])
        state_col = spec["state_col"]
        has_ee = bool(spec.get("has_ee", False))

        a_cnt, a_mean, a_m2 = 0, np.zeros(45), np.zeros(45)
        s_cnt, s_mean, s_m2 = 0, np.zeros(75), np.zeros(75)
        buckets = np.zeros(3, dtype=np.int64)

        for ep in spec["train_episodes"]:
            pq_path = ds_dir / f"data/chunk-000/episode_{int(ep):06d}.parquet"
            cols = ["action", state_col, "valid_for_training", "motion_bucket"]
            tbl = pq.read_table(pq_path, columns=cols)
            mask = tbl["valid_for_training"].to_numpy()
            if not mask.any():
                continue

            b_vals = tbl["motion_bucket"].to_numpy()[mask]
            for bi in range(3):
                buckets[bi] += int((b_vals == bi).sum())

            n_rows = len(mask)
            act_flat = tbl["action"].combine_chunks().flatten().flatten().to_numpy(zero_copy_only=False)
            act_w = len(act_flat) // (n_rows * 10)
            act = act_flat.reshape(n_rows, 10, act_w)[mask]
            a_cnt, a_mean, a_m2 = _streaming_moments_update(a_cnt, a_mean, a_m2, act[:, :, :45].reshape(-1, 45))

            st_flat = tbl[state_col].combine_chunks().flatten().flatten().to_numpy(zero_copy_only=False)
            st_w = len(st_flat) // (n_rows * 4)
            st = st_flat.reshape(n_rows, 4, st_w)[mask]
            s_cnt, s_mean, s_m2 = _streaming_moments_update(s_cnt, s_mean, s_m2, st[:, :, :75].reshape(-1, 75))

            if has_ee and act_w >= 47 and st_w >= 79:
                ee_act_cnt, ee_act_mean, ee_act_m2 = _streaming_moments_update(
                    ee_act_cnt, ee_act_mean, ee_act_m2, act[:, :, 45:47].reshape(-1, 2)
                )
                ee_hist_cnt, ee_hist_mean, ee_hist_m2 = _streaming_moments_update(
                    ee_hist_cnt, ee_hist_mean, ee_hist_m2, st[:, :, 75:79].reshape(-1, 4)
                )

        if a_cnt == 0 or s_cnt == 0:
            raise ValueError(f"Branch {spec['name']} has zero valid train windows")

        per_branch_stats[spec["name"]] = {
            "weight": float(spec["weight"]),
            "train_windows": int(a_cnt // 10),
            "bucket_counts": {
                "locomotion": int(buckets[0]),
                "manipulation": int(buckets[1]),
                "idle": int(buckets[2]),
            },
            "action_5pt_mean": a_mean,
            "action_5pt_var": a_m2 / a_cnt,
            "proprio_5pt_mean": s_mean,
            "proprio_5pt_var": s_m2 / s_cnt,
        }

    # Law of total variance for weighted mixture across active branches
    total_w = sum(b["weight"] for b in per_branch_stats.values())
    if total_w <= 0:
        raise ValueError("Total branch weight must be positive")

    def _mix(key_mean: str, key_var: str, dim: int) -> tuple[np.ndarray, np.ndarray]:
        g_mean = np.zeros(dim, dtype=np.float64)
        for b in per_branch_stats.values():
            g_mean += (b["weight"] / total_w) * b[key_mean]
        g_var = np.zeros(dim, dtype=np.float64)
        for b in per_branch_stats.values():
            w = b["weight"] / total_w
            g_var += w * (b[key_var] + (b[key_mean] - g_mean) ** 2)
        g_std = np.sqrt(np.maximum(g_var, 1e-12))
        return g_mean, g_std

    act_mean, act_std = _mix("action_5pt_mean", "action_5pt_var", 45)
    prop_mean, prop_std = _mix("proprio_5pt_mean", "proprio_5pt_var", 75)

    # EE channels: keep raw [0, 1] / [-1, 1] scale or fitted global EE stats
    ee_a_mean = ee_act_mean if ee_act_cnt > 0 else np.zeros(2)
    ee_a_std = np.sqrt(np.maximum(ee_act_m2 / max(ee_act_cnt, 1), 1e-6)) if ee_act_cnt > 0 else np.ones(2)
    ee_h_mean = ee_hist_mean if ee_hist_cnt > 0 else np.zeros(4)
    ee_h_std = np.sqrt(np.maximum(ee_hist_m2 / max(ee_hist_cnt, 1), 1e-6)) if ee_hist_cnt > 0 else np.ones(4)

    stats = {
        "schema": CANONICAL_SCHEMA,
        "canonical_leg_height_m": CANON_LEG_HEIGHT,
        "canonical_arm_length_m": CANON_ARM_LENGTH,
        "action_5pt": {"mean": act_mean.tolist(), "std": act_std.tolist()},
        "proprio_5pt": {"mean": prop_mean.tolist(), "std": prop_std.tolist()},
        "ee_action": {"mean": ee_a_mean.tolist(), "std": ee_a_std.tolist(), "count": int(ee_act_cnt)},
        "ee_hist": {"mean": ee_h_mean.tolist(), "std": ee_h_std.tolist(), "count": int(ee_hist_cnt)},
        # Convenience full-width keys matching WAMDataset / _FastBranchReader
        "action": {
            "mean": np.concatenate([act_mean, ee_a_mean]).tolist(),
            "std": np.concatenate([act_std, ee_a_std]).tolist(),
        },
        "action_human": {"mean": act_mean.tolist(), "std": act_std.tolist()},
        "observation.state": {"mean": prop_mean.tolist(), "std": prop_std.tolist()},
        "observation.state_5pt_hist": {
            "mean": np.concatenate([prop_mean, ee_h_mean]).tolist(),
            "std": np.concatenate([prop_std, ee_h_std]).tolist(),
        },
        "branches": {
            name: {
                "weight": b["weight"],
                "train_windows": b["train_windows"],
                "bucket_counts": b["bucket_counts"],
            }
            for name, b in per_branch_stats.items()
        },
    }
    write_json(output_path, stats)
    return stats


def check_stereo_seam(video_path: Path, n_samples: int = 8) -> dict[str, Any]:
    """Gate 3: Detect stereo-seam vertical gradient jump at center columns (x=111..112) of 224x224 MP4."""
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(video_path),
        "-vf", f"select='not(mod(n\\,5))'", "-vsync", "vfr", "-frames:v", str(n_samples),
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    raw = np.frombuffer(res.stdout, dtype=np.uint8)
    n_frames = len(raw) // (224 * 224 * 3)
    if n_frames == 0:
        raise ValueError(f"Could not decode frames from {video_path}")
    frames = raw[: n_frames * 224 * 224 * 3].reshape(n_frames, 224, 224, 3).astype(np.float32)
    # Horizontal gradient |I[:, :, x+1] - I[:, :, x]| averaged over frames, height, RGB
    col_grad = np.abs(np.diff(frames, axis=2)).mean(axis=(0, 1, 3))  # [223]
    center_grad = float(col_grad[110:113].max())
    neighbor_grad = float(np.median(np.concatenate([col_grad[80:105], col_grad[118:143]])))
    ratio = center_grad / max(neighbor_grad, 1.0)
    return {
        "frames_checked": int(n_frames),
        "center_grad": round(center_grad, 3),
        "neighbor_median_grad": round(neighbor_grad, 3),
        "center_to_neighbor_ratio": round(ratio, 3),
        "seam_detected": bool(ratio > 3.0 and center_grad > 20.0),
    }


def make_rgb_contact_sheet(
    dataset_dir: Path,
    episode_ids: list[int],
    output_png: Path,
    *,
    num_windows: int = 20,
    seed: int = 20261123,
) -> dict[str, Any]:
    """Generate a 4x5 RGB contact sheet (20 random windows) for visual QA gate."""
    from PIL import Image, ImageDraw

    rng = np.random.default_rng(seed)
    chosen = rng.choice(episode_ids, size=min(num_windows, len(episode_ids)), replace=len(episode_ids) < num_windows)
    tiles = []
    seam_flags = 0
    max_seam_ratio = 0.0
    for ep in chosen:
        vid = dataset_dir / f"videos/chunk-000/observation.images.cam_head/episode_{int(ep):06d}.mp4"
        seam = check_stereo_seam(vid, n_samples=4)
        if seam["seam_detected"]:
            seam_flags += 1
        max_seam_ratio = max(max_seam_ratio, seam["center_to_neighbor_ratio"])
        cmd = [
            "ffmpeg", "-v", "error", "-i", str(vid),
            "-vf", "select='eq(n\\,10)'", "-vframes", "1",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        if len(res.stdout) < 224 * 224 * 3:
            cmd[5] = "select='eq(n\\,0)'"
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        arr = np.frombuffer(res.stdout[: 224 * 224 * 3], dtype=np.uint8).reshape(224, 224, 3)
        img = Image.fromarray(arr)
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, 120, 16], fill=(0, 0, 0))
        draw.text((3, 2), f"ep{int(ep):05d} r={seam['center_to_neighbor_ratio']:.1f}", fill=(255, 255, 0))
        tiles.append(img)

    cols, rows = 5, 4
    sheet = Image.new("RGB", (cols * 224, rows * 224), (20, 20, 20))
    for i, img in enumerate(tiles[: cols * rows]):
        sheet.paste(img, ((i % cols) * 224, (i // cols) * 224))
    output_png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_png)
    return {
        "contact_sheet": str(output_png),
        "samples": len(tiles),
        "seam_anomalies": seam_flags,
        "max_seam_ratio": round(max_seam_ratio, 3),
    }


def run_six_gate_qa(
    branch_name: str,
    branch_root: Path,
    dataset_name: str,
    *,
    state_col: str,
    global_stats_path: Path | None,
    output_report_path: Path,
    contact_sheet_path: Path,
    sample_episodes: int = 32,
) -> dict[str, Any]:
    """Run all 6 mandatory QA gates from docs/design/data-processing.md §1.5."""
    ds_dir = Path(branch_root) / "wam_lerobot_v2" / dataset_name
    splits = json.loads((ds_dir / "meta/splits.json").read_text())
    episodes = {
        r["episode_index"]: r
        for r in map(json.loads, (ds_dir / "meta/episodes.jsonl").read_text().splitlines())
    }

    # Gate 5: Split isolation (no overlap between train / val / test)
    all_split_eps = [ep for eps in splits.values() for ep in eps]
    gate5_pass = len(all_split_eps) == len(set(all_split_eps))

    train_eps = splits.get("train", [])
    eval_eps = train_eps if train_eps else all_split_eps
    rng = np.random.default_rng(20261123)
    probe_eps = sorted(rng.choice(eval_eps, size=min(sample_episodes, len(eval_eps)), replace=False).tolist())

    t0_poses_list = []
    act_rot_err_max = 0.0
    nonfinite_count = 0
    bucket_counts = np.zeros(3, dtype=np.int64)
    total_valid = 0

    for ep in probe_eps:
        tbl = pq.read_table(
            ds_dir / f"data/chunk-000/episode_{ep:06d}.parquet",
            columns=["action", state_col, "valid_for_training", "motion_bucket"],
        )
        mask = tbl["valid_for_training"].to_numpy()
        if not mask.any():
            continue
        total_valid += int(mask.sum())
        b_vals = tbl["motion_bucket"].to_numpy()[mask]
        for bi in range(3):
            bucket_counts[bi] += int((b_vals == bi).sum())

        n_rows = len(mask)
        act_flat = tbl["action"].combine_chunks().flatten().flatten().to_numpy(zero_copy_only=False)
        act_w = len(act_flat) // (n_rows * 10)
        act = act_flat.reshape(n_rows, 10, act_w)[mask]

        st_flat = tbl[state_col].combine_chunks().flatten().flatten().to_numpy(zero_copy_only=False)
        st_w = len(st_flat) // (n_rows * 4)
        st = st_flat.reshape(n_rows, 4, st_w)[mask]

        if not (np.isfinite(act).all() and np.isfinite(st).all()):
            nonfinite_count += 1

        # t0 poses (last history frame, first 45D)
        t0_poses_list.append(st[:, -1, :45])

        # rot6d orthogonality residual
        r6 = act[:, :, 15:45].reshape(-1, 5, 2, 3)
        norm_err = np.abs(np.linalg.norm(r6, axis=-1) - 1.0).max()
        dot_err = np.abs((r6[:, :, 0] * r6[:, :, 1]).sum(axis=-1)).max()
        act_rot_err_max = max(act_rot_err_max, float(max(norm_err, dot_err)))

    t0_poses = np.concatenate(t0_poses_list, axis=0)
    pos_med = np.median(t0_poses[:, :15].reshape(-1, 5, 3), axis=0)

    # Gate 1: Coordinate axis & sign checks
    la_z, ra_z = float(pos_med[1, 2]), float(pos_med[2, 2])
    la_y, ra_y = float(pos_med[1, 1]), float(pos_med[2, 1])
    lw_y, rw_y = float(pos_med[3, 1]), float(pos_med[4, 1])
    wri_x_med = float(0.5 * (pos_med[3, 0] + pos_med[4, 0]))
    gate1_checks = {
        "L_ank_z_in_range": -1.10 <= la_z <= -0.85,
        "R_ank_z_in_range": -1.10 <= ra_z <= -0.85,
        "L_ank_y_pos": la_y > 0.05,
        "R_ank_y_neg": ra_y < -0.05,
        "L_wri_y_pos": lw_y > 0.0,
        "R_wri_y_neg": rw_y < 0.0,
        "wri_x_pos": wri_x_med > 0.0,
    }
    gate1_pass = all(gate1_checks.values())

    # Gate 2: Round-trip precision Canonical Human -> Target Robot (Q1 & G1) -> Canonical Human
    sample_45 = t0_poses[: min(64, len(t0_poses))]
    anchor_dummy = np.zeros((len(sample_45), 5, 7), dtype=np.float64)
    anchor_dummy[:, 0, 2] = 0.85
    anchor_dummy[:, :, 3] = 1.0  # identity wxyz
    rt_max_pos_mm = 0.0
    rt_max_rot_deg = 0.0
    for robot_name in ("q1", "g1"):
        geom = ROBOT_GEOMETRY[robot_name]
        w_robot = canonical_5pt_to_target_world(sample_45, anchor_dummy, geom["leg_height"], geom["arm_length"])
        rec_45 = world_flu_to_canonical_5pt(w_robot, anchor_dummy, geom["leg_height"], geom["arm_length"])
        pos_diff = np.abs(rec_45[:, :15] - sample_45[:, :15]).reshape(-1, 5, 3)
        pos_diff[:, :3, :] *= CANON_LEG_HEIGHT
        rt_max_pos_mm = max(rt_max_pos_mm, float(pos_diff.max() * 1000.0))
        r_orig = rotation6d(sample_45[:, 15:].reshape(-1, 5, 6))
        r_rec = rotation6d(rec_45[:, 15:].reshape(-1, 5, 6))
        ang_rad = Rotation.from_matrix((r_orig.swapaxes(-1, -2) @ r_rec).reshape(-1, 3, 3)).magnitude()
        rt_max_rot_deg = max(rt_max_rot_deg, float(np.degrees(ang_rad.max())))
    gate2_pass = (
        rt_max_pos_mm < 1.0
        and rt_max_rot_deg < 0.1
        and act_rot_err_max < 1e-4
        and nonfinite_count == 0
    )

    # Gate 3: Stereo seam detection + 20-window RGB contact sheet
    visual_qa = make_rgb_contact_sheet(ds_dir, all_split_eps, contact_sheet_path, num_windows=20)
    gate3_pass = visual_qa["seam_anomalies"] == 0

    # Gate 4: Causality check (latent frames >= 3 and <= length - 13, verified in cache receipts)
    from safetensors import safe_open
    gate4_pass = True
    for ep in probe_eps[:8]:
        lat_path = ds_dir / f"latents/chunk-000/episode_{ep:06d}.safetensors"
        if not lat_path.is_file():
            gate4_pass = False
            break
        with safe_open(str(lat_path), framework="pt", device="cpu") as sf:
            f_idx = sf.get_tensor("frame_index").numpy()
        if (f_idx < 3).any() or (f_idx + 12 >= episodes[ep]["length"]).any():
            gate4_pass = False
            break

    # Gate 6: Global stats z-score sanity check (median standing pose within [-1.5 sigma, +1.5 sigma])
    gate6_pass = True
    gate6_max_abs_z = None
    if global_stats_path is not None and Path(global_stats_path).is_file():
        gstats = json.loads(Path(global_stats_path).read_text())
        p_mean = np.asarray(gstats["proprio_5pt"]["mean"][:45], dtype=np.float64)
        p_std = np.maximum(np.asarray(gstats["proprio_5pt"]["std"][:45], dtype=np.float64), 1e-6)
        z_med = (np.median(t0_poses, axis=0) - p_mean) / p_std
        gate6_max_abs_z = round(float(np.abs(z_med).max()), 3)
        gate6_pass = gate6_max_abs_z <= 1.5

    report = {
        "schema": CANONICAL_SCHEMA,
        "branch": branch_name,
        "dataset": dataset_name,
        "ready": bool(gate1_pass and gate2_pass and gate3_pass and gate4_pass and gate5_pass and gate6_pass),
        "episodes_total": len(episodes),
        "probe_episodes": len(probe_eps),
        "probe_valid_windows": total_valid,
        "probe_bucket_counts": {
            "locomotion": int(bucket_counts[0]),
            "manipulation": int(bucket_counts[1]),
            "idle": int(bucket_counts[2]),
        },
        "gates": {
            "gate1_axes_and_signs": {
                "pass": gate1_pass,
                "checks": gate1_checks,
                "medians": {
                    "L_ank_z": round(la_z, 4),
                    "R_ank_z": round(ra_z, 4),
                    "L_ank_y": round(la_y, 4),
                    "R_ank_y": round(ra_y, 4),
                    "L_wri_y": round(lw_y, 4),
                    "R_wri_y": round(rw_y, 4),
                    "wri_x_median": round(wri_x_med, 4),
                },
            },
            "gate2_roundtrip_and_orthogonality": {
                "pass": gate2_pass,
                "max_pos_err_mm": round(rt_max_pos_mm, 6),
                "max_rot_err_deg": round(rt_max_rot_deg, 6),
                "max_rot6d_orth_err": round(act_rot_err_max, 8),
                "nonfinite_episodes": nonfinite_count,
            },
            "gate3_visual_and_seam": {
                "pass": gate3_pass,
                **visual_qa,
            },
            "gate4_causality": {"pass": gate4_pass},
            "gate5_split_isolation": {
                "pass": gate5_pass,
                "split_counts": {k: len(v) for k, v in splits.items()},
            },
            "gate6_global_stats_alignment": {
                "pass": gate6_pass,
                "max_abs_z_median": gate6_max_abs_z,
            },
        },
    }
    write_json(output_report_path, report)
    return report
