"""Unit tests for Canonical Human (H_5') five-point v2.0 geometry and QA."""
import numpy as np
from scipy.spatial.transform import Rotation

from wammimic.data_pipeline.canonical_human import (
    BUCKET_IDLE,
    BUCKET_LOCOMOTION,
    BUCKET_MANIPULATION,
    CANON_ARM_LENGTH,
    CANON_LEG_HEIGHT,
    C_NOKOV_TO_FLU,
    ROBOT_GEOMETRY,
    canonical_5pt_to_target_world,
    compute_motion_metrics,
    reproject_old_history_to_canonical,
    world_flu_to_canonical_5pt,
    world_flu_to_canonical_history,
)
from wammimic.data_pipeline.geometry import rotation6d
from wammimic.data_pipeline.human_ego import human_history


def _standing_poses(leg_h=0.85, arm_l=0.58, yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0):
    """Build a synthetic standing 5-point world FLU pose [5, 7]."""
    r_yaw = Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix()
    r_pelvis = Rotation.from_euler("ZYX", [yaw_deg, pitch_deg, roll_deg], degrees=True).as_matrix()
    p0 = np.array([1.2, -0.5, leg_h + 0.05])
    rel_local = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.12 * (leg_h / 0.85), -leg_h],
        [0.0, -0.12 * (leg_h / 0.85), -leg_h],
        [0.30 * (arm_l / 0.58), 0.20 * (arm_l / 0.58), 0.10],
        [0.30 * (arm_l / 0.58), -0.20 * (arm_l / 0.58), 0.10],
    ])
    pos = p0[None, :] + (r_yaw @ rel_local.T).T
    q_pelvis = Rotation.from_matrix(r_pelvis).as_quat()[[3, 0, 1, 2]]
    q_yaw = Rotation.from_matrix(r_yaw).as_quat()[[3, 0, 1, 2]]
    quats = np.stack([q_pelvis, q_yaw, q_yaw, q_yaw, q_yaw], axis=0)
    return np.concatenate([pos, quats], axis=-1).astype(np.float32)


def test_yaw_only_decouples_pitch_and_roll():
    """Bending forward by 35 deg must NOT swing ankles up/backwards in {P_t0}."""
    p_upright = _standing_poses(leg_h=0.745, arm_l=0.50, yaw_deg=40.0, pitch_deg=0.0, roll_deg=0.0)
    p_bowed = _standing_poses(leg_h=0.745, arm_l=0.50, yaw_deg=40.0, pitch_deg=35.0, roll_deg=-15.0)
    f_up = world_flu_to_canonical_5pt(p_upright, p_upright, 0.745, 0.50)
    f_bow = world_flu_to_canonical_5pt(p_bowed, p_bowed, 0.745, 0.50)

    pos_up = f_up[:15].reshape(5, 3)
    pos_bow = f_bow[:15].reshape(5, 3)
    # Ankle z stays strictly -1.00 even when torso pitches 35 deg!
    assert np.allclose(pos_up[1:3, 2], -1.0, atol=1e-4)
    assert np.allclose(pos_bow[1:3, 2], -1.0, atol=1e-4)
    # Wrist positions mapped to Canonical Human scale (0.30, +-0.20)
    assert np.allclose(pos_up[3, :2], [0.30, 0.20], atol=1e-4)
    assert np.allclose(pos_up[4, :2], [0.30, -0.20], atol=1e-4)
    # Pelvis pitch/roll is preserved in pelvis rot6d
    r_pelvis_bow = rotation6d(f_bow[15:21])
    expected_pr = Rotation.from_euler("YX", [35.0, -15.0], degrees=True).as_matrix()
    assert np.allclose(r_pelvis_bow, expected_pr, atol=1e-4)


def test_roundtrip_canonical_to_q1_and_g1():
    """Canonical Human -> Q1 / G1 -> Canonical Human must be < 1 mm and < 0.1 deg."""
    rng = np.random.default_rng(42)
    anchor = _standing_poses(leg_h=0.85, arm_l=0.58, yaw_deg=-75.0, pitch_deg=12.0)
    future = np.stack([_standing_poses(leg_h=0.85, arm_l=0.58, yaw_deg=-75.0 + 3 * t) for t in range(1, 11)], axis=0)
    future[:, :, :3] += rng.normal(0, 0.05, size=(10, 5, 3))
    c45 = world_flu_to_canonical_5pt(future, anchor, 0.85, 0.58)

    for robot in ("q1", "g1"):
        g = ROBOT_GEOMETRY[robot]
        w_robot = canonical_5pt_to_target_world(c45, anchor, g["leg_height"], g["arm_length"])
        rec45 = world_flu_to_canonical_5pt(w_robot, anchor, g["leg_height"], g["arm_length"])
        pos_err = np.abs(rec45[:, :15] - c45[:, :15]).reshape(10, 5, 3)
        pos_err[:, :3, :] *= CANON_LEG_HEIGHT
        assert pos_err.max() < 1e-3  # < 1 mm
        r_a = rotation6d(c45[:, 15:].reshape(-1, 5, 6))
        r_b = rotation6d(rec45[:, 15:].reshape(-1, 5, 6))
        deg = np.degrees(Rotation.from_matrix((r_a.swapaxes(-1, -2) @ r_b).reshape(-1, 3, 3)).magnitude().max())
        assert deg < 0.1


def test_nokov_calibration_and_reproject_history():
    """NOKOV local axes (X=Left, Y=Up, Z=Fwd) map cleanly to FLU and reprojected history matches exact."""
    t = np.arange(20, dtype=np.float64) * 0.1
    poses_flu = np.stack([_standing_poses(leg_h=0.78, arm_l=0.52, yaw_deg=20 + 2 * i, pitch_deg=5 + i) for i in range(20)])
    poses_flu[:, :, 0] += t[:, None] * 0.3

    # Convert FLU rotations to NOKOV local convention: R_nokov = R_flu @ C_NOKOV_TO_FLU.T
    r_flu = Rotation.from_quat(poses_flu[:, :, 3:].reshape(-1, 4)[:, [1, 2, 3, 0]]).as_matrix()
    r_nokov = r_flu @ C_NOKOV_TO_FLU.T
    q_nokov = Rotation.from_matrix(r_nokov).as_quat()[:, [3, 0, 1, 2]].reshape(20, 5, 4)
    poses_nokov = poses_flu.copy()
    poses_nokov[:, :, 3:] = q_nokov

    valid_frames = np.array([4, 5, 6, 7], dtype=np.int64)
    hist_idx = valid_frames[:, None] + np.arange(-3, 1)

    # Exact canonical history from NOKOV poses
    exact_hist = world_flu_to_canonical_history(poses_nokov, t, hist_idx, 0.78, 0.52, c_rest=C_NOKOV_TO_FLU)

    # Simulate old human_history (full 3D pelvis, old leg=0.80) and reproject
    old_state = np.zeros((20, 4, 75), dtype=np.float32)
    old_state[valid_frames] = human_history(poses_nokov, t, hist_idx, 0.80)
    reproj = reproject_old_history_to_canonical(
        poses_nokov, old_state, valid_frames, 0.80, 0.78, 0.52, c_rest=C_NOKOV_TO_FLU
    )
    assert np.allclose(reproj[valid_frames], exact_hist, atol=1e-4)


def test_motion_buckets():
    """Verify Bucket A (locomotion), Bucket B (manipulation), Bucket C (idle)."""
    base = np.zeros((3, 45), dtype=np.float32)
    base[:, 5] = -1.0  # LA z
    base[:, 8] = -1.0  # RA z
    act = np.repeat(base[:, None, :], 10, axis=1)
    # Sample 0: walk forward 0.30m -> Bucket A
    act[0, -1, 0] = 0.30 / CANON_LEG_HEIGHT
    # Sample 1: reach wrist forward 0.25m while feet still -> Bucket B
    act[1, 5, 9] = 0.25
    # Sample 2: idle -> Bucket C
    p_d, a_l, w_d, b = compute_motion_metrics(act, base)
    assert b.tolist() == [BUCKET_LOCOMOTION, BUCKET_MANIPULATION, BUCKET_IDLE]
    assert abs(p_d[0] - 0.30) < 1e-4
    assert abs(w_d[1] - 0.25) < 1e-4
