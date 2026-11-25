"""Evaluate WAM + Mimic on an in-distribution OpenHLM tabletop scene in Genie Sim (Q1 & G1).

Scene matches OpenHLM (`gripper`, embodiment_id=1) task_0:
  "Walk forward to the table, put the red can on the orange mouse pad with the right hand, turn around and walk back."
Uses Genie Sim's published `table_000` and `benchmark_coaster_015` (colored orange) plus a red can cylinder,
and loads the exact pre-encoded Cosmos-Reason1 `task_0` prompt embedding from the OpenHLM training branch.
"""
import argparse
from collections import deque
from contextlib import ExitStack
import json
from pathlib import Path
import subprocess
from time import perf_counter

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation, Slerp
from safetensors import safe_open
import torch

from morphmimic.constants import GOAL_K, HUMAN_FIVE_IDX, NUM_HUMAN, PROPRIO_DIM
from morphmimic.env.env_cfg import COMMAND_NAME, morphmimic_env_cfg
from morphmimic.eval.policy_loader import load_policy
from morphmimic.model.human import pack_human_goal
from morphmimic.model.ref_feedback import REF_FRAMES, REF_SLOTS, global_ref_feedback
from morphmimic.model.state import anchor_quat, join_state, split_state
from morphmimic.train.obs_history import ObsHistory
from preview.cpu_backend import CpuPreviewEnv
from scripts.evaluate_geniesim_wam_mimic import (
    DT,
    SLOTS,
    actor_targets,
    interpolate,
    predict,
    render_panel,
    usd_geometry,
    wxyz,
)
from wammimic.data_pipeline.cache import VideoEncoder
from wammimic.data_pipeline.common import sha256, write_json
from wammimic.data_pipeline.geometry import LEG_LENGTH, matrices
from wammimic.data_pipeline.human_ego import human_history
from wammimic.training.evaluation import rotation6d
from wammimic.training.model import CosmosWAM, load_backbone


SCALE_Q1 = LEG_LENGTH["q1"] / LEG_LENGTH["g1"]


def build_red_can_scene(assets: Path, *, scale: float, robot: str, fovy: float = 72.0):
    """Build Genie Sim white table + orange mouse pad + red can matching OpenHLM task_0."""
    table_meshes = usd_geometry(assets / "objects/edited/table_000/Aligned.usda")
    coaster_meshes = usd_geometry(assets / "objects/benchmark/coaster/benchmark_coaster_015/Aligned.usda")

    table_xyz = np.array([0.0, 0.0, 0.0], dtype=float)
    source_top = table_xyz[2] + max(pts[:, 2].max() for _, pts, _ in table_meshes)
    top = source_top * scale

    coaster_bottom = min(pts[:, 2].min() for _, pts, _ in coaster_meshes)
    can_radius = 0.033 * (scale if robot == "q1" else 1.0)
    can_half_h = 0.058 * (scale if robot == "q1" else 1.0)

    def configure(spec):
        torso = spec.body("robot/torso_link")
        if robot == "q1":
            pitch = np.deg2rad(27.0)
            cam_pos = [0.085, 0.0, 0.23]
        else:
            # G1 head camera height & slight downward pitch matching OpenHLM ego view
            pitch = np.deg2rad(26.0)
            cam_pos = [0.10, 0.0, 0.38]
        torso.add_camera(
            name="wam_head",
            pos=cam_pos,
            fovy=fovy,
            xyaxes=[0.0, -1.0, 0.0, np.sin(pitch), 0.0, np.cos(pitch)],
        )

        # White table (Genie Sim table_000)
        table_body = spec.worldbody.add_body(name="table_openhlm", pos=table_xyz * scale)
        for idx, (_, pts, faces) in enumerate(table_meshes):
            mname = f"table_openhlm_{idx}"
            spec.add_mesh(name=mname, uservert=(pts * scale).ravel(), userface=faces.ravel())
            table_body.add_geom(
                name=mname,
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=mname,
                rgba=(0.88, 0.87, 0.84, 1.0),
                mass=1.0,
                contype=0,
                conaffinity=0,
            )
        spec.worldbody.add_geom(
            name="genie_table_collision",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0.0, 0.0, top - 0.025],
            size=[0.30 * scale, 0.50 * scale, 0.025],
            rgba=[0.0, 0.0, 0.0, 0.0],
            contype=1,
            conaffinity=1,
        )

        # Orange mouse pad (Genie Sim coaster scaled slightly wider, placed left-center on table)
        pad_scale = np.array([1.35 * scale, 1.35 * scale, 1.0])
        pad_pos = np.array([-0.04 * scale, 0.16 * scale, top - coaster_bottom + 0.001])
        pad_body = spec.worldbody.add_body(name="orange_mouse_pad", pos=pad_pos)
        for idx, (_, pts, faces) in enumerate(coaster_meshes):
            mname = f"orange_pad_{idx}"
            spec.add_mesh(name=mname, uservert=(pts * pad_scale).ravel(), userface=faces.ravel())
            pad_body.add_geom(
                name=mname,
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=mname,
                rgba=(0.95, 0.45, 0.10, 1.0),
                mass=0.05,
                contype=1,
                conaffinity=1,
            )

        # Red soda can on right-center of table (OpenHLM task_0 target object)
        can_pos = np.array([-0.06 * scale, -0.06 * scale, top + can_half_h + 0.001])
        can_body = spec.worldbody.add_body(name="red_can", pos=can_pos)
        can_body.add_geom(
            name="task_target",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[can_radius, can_half_h, 0.0],
            rgba=(0.86, 0.12, 0.14, 1.0),
            mass=0.15,
            contype=1,
            conaffinity=1,
        )

    return configure


def prepare_reference(folder: Path, data: Path, robot: str, spawn: np.ndarray, frames: int):
    folder.mkdir(parents=True, exist_ok=True)
    with np.load(data / f"{robot}/sprint1_subject2.npz") as original:
        arrays = {
            k: np.repeat(original[k][:1], frames, axis=0) if original[k].ndim > 1 else original[k]
            for k in original.files
        }
        arrays["body_pos_w"] += spawn
        for name in ("joint_vel", "body_lin_vel_w", "body_ang_vel_w"):
            arrays[name].fill(0.0)
        np.savez(folder / f"neutral_{robot}.npz", **arrays)
    with np.load(data / f"human50/{robot}/sprint1_subject2.npz") as human:
        human_pos = human["human_pos_w"][0, list(HUMAN_FIVE_IDX)].copy()
        human_rot = matrices(human["human_quat_w"][0, list(HUMAN_FIVE_IDX)])
    subset = folder / f"subset_{robot}.yaml"
    write_json(subset, {"root": str(folder), "clips": [f"neutral_{robot}.npz"]})
    return (
        subset,
        (
            arrays["body_pos_w"][0, list(SLOTS)],
            matrices(arrays["body_quat_w"][0, list(SLOTS)]),
            human_pos,
            human_rot,
        ),
    )


def targets_from_wam(
    block: np.ndarray,
    actual: tuple[np.ndarray, np.ndarray],
    rests: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    robot: str,
    bridge: str,
    act_mean: np.ndarray,
    anchor_xy: np.ndarray | None = None,
    max_lead_m: float = 0.08,
):
    """Map WAM 10x47 output to Mimic human & robot 5-point targets.

    bridge == 'raw': exact reproduction of q1_direct_targets() (full-3D r0, no arm/ankle retargeting).
    bridge == 'yaw_retarget': yaw-only heading unprojection (breaks pitch bowing loop) + G1->Q1 arm/ankle calibration.
    """
    robot_rest_pos, robot_rest_rot, human_rest_pos, human_rest_rot = rests
    current_pos, current_rot = actual
    leg = LEG_LENGTH[robot]
    pos_local = block[:, :15].reshape(-1, 5, 3).copy()
    rot_local, valid = rotation6d(block[:, 15:45].reshape(-1, 5, 6))
    if not np.isfinite(pos_local).all() or not valid.all():
        raise ValueError("Nonfinite or degenerate WAM target")

    origin_xy = current_pos[0, :2].copy()
    if anchor_xy is not None:
        lead = anchor_xy - origin_xy
        norm = float(np.linalg.norm(lead))
        if norm > max_lead_m:
            lead = lead * (max_lead_m / norm)
        origin_xy = origin_xy + lead
    origin = np.array([origin_xy[0], origin_xy[1], robot_rest_pos[0, 2]])

    if bridge == "raw":
        r0 = current_rot[0]
        pos_local[:, :3] *= leg
        goal = pos_local @ r0.T + origin
        goal_rot = r0[None, None] @ rot_local
    elif bridge == "yaw_retarget":
        yaw = float(np.arctan2(current_rot[0, 1, 0], current_rot[0, 0, 0]))
        heading = Rotation.from_euler("z", yaw).as_matrix()
        # Calibrate G1 normalized ankle rest Z (-1.094) to target robot's true standing ankle-pelvis Z
        mean_pos = act_mean[:15].reshape(5, 3)
        if robot == "q1":
            arm_scale = SCALE_Q1
            rest_ank_z_norm = float(np.mean(robot_rest_pos[1:3, 2] - robot_rest_pos[0, 2])) / leg
            g1_ank_z_norm = float(np.mean(mean_pos[1:3, 2]))
            pos_local[:, 1:3, 2] += (rest_ank_z_norm - g1_ank_z_norm)
            pos_local[:, :3] *= leg
            pos_local[:, 3:] *= arm_scale
        else:
            # Native G1: calibrate any tiny MuJoCo standing ankle Z offset vs OpenHLM FK mean
            rest_ank_z_norm = float(np.mean(robot_rest_pos[1:3, 2] - robot_rest_pos[0, 2])) / leg
            g1_ank_z_norm = float(np.mean(mean_pos[1:3, 2]))
            pos_local[:, 1:3, 2] += (rest_ank_z_norm - g1_ank_z_norm)
            pos_local[:, :3] *= leg
        # Keep ankle targets above floor safety floor (robot standing ankle height - 1 cm)
        goal = pos_local @ heading.T + origin
        min_ank_z = float(min(robot_rest_pos[1, 2], robot_rest_pos[2, 2])) - 0.005
        goal[:, 1:3, 2] = np.maximum(goal[:, 1:3, 2], min_ank_z)
        goal_rot = heading[None, None] @ rot_local
    else:
        raise ValueError(f"Unknown bridge: {bridge}")

    human_scale = 1.65 if robot == "q1" else 1.07
    spawn_xy = np.array([robot_rest_pos[0, 0], robot_rest_pos[0, 1], 0.0])
    human = (human_rest_pos + spawn_xy)[None] + (goal - robot_rest_pos[None]) * human_scale
    human_rot = goal_rot @ robot_rest_rot.swapaxes(-1, -2) @ human_rest_rot
    return (human, human_rot), (goal, goal_rot)


def retarget_proprio_for_g1_stats(h75: np.ndarray, rests, robot: str, s_mean: np.ndarray) -> np.ndarray:
    """Align Q1 4x75 proprio into G1's OpenHLM metric/normalized space before z-scoring."""
    if robot != "q1":
        return h75
    out = h75.copy()
    leg = LEG_LENGTH["q1"]
    rest_ank_z_norm = float(np.mean(rests[0][1:3, 2] - rests[0][0, 2])) / leg
    g1_ank_z_norm = float(np.mean([s_mean[5], s_mean[8]]))
    # Shift normalized ankle Z (indices 5 and 8 of 45D pose) to G1's standing baseline
    out[:, 5] += (g1_ank_z_norm - rest_ank_z_norm)
    out[:, 8] += (g1_ank_z_norm - rest_ank_z_norm)
    # Scale wrist positions (indices 9:15) and wrist linear velocities (indices 54:60) from Q1 m -> G1 m
    out[:, 9:15] /= SCALE_Q1
    out[:, 54:60] /= SCALE_Q1
    return out


@torch.inference_mode()
def run_episode(
    env,
    policy,
    meta,
    wam,
    vae,
    text,
    act_stats,
    st_stats,
    rests,
    *,
    robot: str,
    bridge: str,
    scenario: str,
    mode_label: str,
    seed: int,
    video_path: Path,
    steps: int,
    emb_id: int = 1,
):
    state_ids = torch.arange(1, device=env.device)
    command = env.command_manager.get_term(COMMAND_NAME)
    command.assign_clips(state_ids, torch.zeros_like(state_ids))
    obs, _ = env.reset(env_ids=state_ids)
    history = ObsHistory(1, meta.hist_len, PROPRIO_DIM, env.device)
    history.reset(state_ids, obs["policy"])
    generator = torch.Generator(device="cuda:0").manual_seed(seed)
    model = env.sim.mj_model
    data = env.native_data
    model.vis.global_.offwidth, model.vis.global_.offheight = 960, 540
    camera = mujoco.MjvCamera()
    if robot == "q1":
        camera.lookat[:] = [-0.25, 0.0, 0.34]
        camera.distance, camera.azimuth, camera.elevation = 1.62, 128.0, -18.0
    else:
        camera.lookat[:] = [-0.35, 0.0, 0.62]
        camera.distance, camera.azimuth, camera.elevation = 2.45, 128.0, -18.0
    target = model.geom("task_target").id
    hand_geoms = {
        g for g in range(model.ngeom)
        if "robot/" in model.body(model.geom_bodyid[g]).name
        and any(term in model.body(model.geom_bodyid[g]).name for term in ("wrist", "hand"))
    }
    frames = deque(maxlen=4)
    updates, contact_steps = [], 0
    closest, reason = np.inf, "timeout"
    human = robot_tgt = None
    spawn_xy = np.array([rests[0][0, 0], rests[0][0, 1], 0.0])
    human_start = rests[2] + spawn_xy
    trajectories = []
    mean = np.asarray(act_stats["mean"], dtype=np.float32)
    scale = np.maximum(np.asarray(act_stats["std"], dtype=np.float32), 1e-6)
    s_mean = np.asarray(st_stats["mean"], dtype=np.float32)
    s_scale = np.maximum(np.asarray(st_stats["std"], dtype=np.float32), 1e-6)
    poses_50hz = []

    with ExitStack() as stack:
        head = stack.enter_context(mujoco.Renderer(model, height=224, width=224))
        wide = stack.enter_context(mujoco.Renderer(model, height=540, width=960))
        video = stack.enter_context(subprocess.Popen(
            [
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "960x540", "-r", "25",
                "-i", "pipe:0", "-c:v", "libx264", "-threads", "2", "-pix_fmt", "yuv420p",
                str(video_path),
            ],
            stdin=subprocess.PIPE,
        ))
        last_image = None
        for k in range(steps):
            state = env.state()
            parts = split_state(state)
            actual_pos = parts.body_pos[0, list(SLOTS)].cpu().numpy()
            actual_quat = parts.body_quat[0, list(SLOTS)].cpu().numpy()
            actual_rot = matrices(actual_quat)
            poses_50hz.append(np.concatenate((actual_pos, actual_quat), axis=-1))

            if k % 5 == 0:
                head.update_scene(data, camera="wam_head")
                last_image = head.render().copy()
                frames.append(last_image)

            if len(frames) == 4 and (k - 15) % 10 == 0:
                started = perf_counter()
                old = k - updates[-1]["control_step"] if updates else 0
                anchor_xy = None if robot_tgt is None else robot_tgt[0][old, 0, :2]
                p_arr = np.asarray(poses_50hz, dtype=np.float64)
                t_arr = np.arange(len(p_arr), dtype=np.float64) * DT
                hist_idx = np.maximum(np.array([[k - 15, k - 10, k - 5, k]], dtype=int), 1)
                h75 = human_history(p_arr, t_arr, hist_idx, LEG_LENGTH[robot])[0]
                if bridge == "yaw_retarget":
                    h75 = retarget_proprio_for_g1_stats(h75, rests, robot, s_mean)
                st75 = (h75 - s_mean[:75]) / s_scale[:75]
                ee_h4 = np.zeros((4, 4), dtype=np.float32)
                prediction = predict(
                    wam, vae, list(frames), text, generator, mean, scale,
                    proprio=st75.astype(np.float32), ee_hist=ee_h4, embodiment_id=emb_id,
                )
                human_block, robot_block = targets_from_wam(
                    prediction, (actual_pos, actual_rot), rests,
                    robot=robot, bridge=bridge, act_mean=mean, anchor_xy=anchor_xy,
                )
                start_h = (human_start, rests[3]) if human is None else (human[0][old], human[1][old])
                start_r = (actual_pos, actual_rot) if robot_tgt is None else (robot_tgt[0][old], robot_tgt[1][old])
                human, robot_tgt = interpolate(start_h, human_block), interpolate(start_r, robot_block)
                updates.append({"control_step": k, "inference_s": perf_counter() - started})

            if robot_tgt is None:
                targets = actor_targets(
                    state,
                    (np.repeat(human_start[None], 11, 0), np.repeat(rests[3][None], 11, 0)),
                    (np.repeat(rests[0][None], 11, 0), np.repeat(rests[1][None], 11, 0)),
                    0,
                )
                markers = None
            else:
                index = k - updates[-1]["control_step"]
                targets = actor_targets(state, human, robot_tgt, index)
                markers = robot_tgt[0][index]

            pairs = [(int(c.geom[0]), int(c.geom[1])) for c in data.contact[:data.ncon] if c.dist <= 0.001]
            contact = any((a == target and b in hand_geoms) or (b == target and a in hand_geoms) for a, b in pairs)
            contact_steps += int(contact)
            distance = float(np.linalg.norm(actual_pos[3:] - data.geom_xpos[target], axis=-1).min())
            closest = min(closest, distance)
            action = policy(history.flat(), targets)
            if not bool(torch.isfinite(action).all()):
                raise ValueError("Mimic produced nonfinite action")

            trajectories.append({
                "step": k,
                "wrist_target_distance_m": distance,
                "hand_contact": contact,
                "task_object_position_m": data.geom_xpos[target].tolist(),
                "target_positions_m": None if markers is None else markers.tolist(),
                "actual_positions_m": actual_pos.tolist(),
            })
            reached = contact_steps > 0
            if k % 2 == 0 or reached:
                label = f"{scenario} / {mode_label} / seed {seed}   {k * DT:.2f}s   closest {closest*100:.1f}cm   contact {contact_steps}"
                pixels = render_panel(wide, data, camera, last_image, markers, label).tobytes()
                video.stdin.write(pixels * (13 if reached else 1))
            if reached:
                reason = "success"
                break
            obs, _, ended, truncated, _ = env.step(action)
            history.push(obs["policy"])
            if bool(ended[0]) or bool(truncated[0]):
                terms = env.termination_manager
                reason = "+".join(name for name in terms.active_terms if bool(terms.get_term(name)[0])) or "native_termination"
                break
        video.stdin.close()
        if video.wait() != 0:
            raise RuntimeError("Replay encoding failed")

    video_path.with_suffix(".jsonl").write_text("".join(json.dumps(r) + "\n" for r in trajectories))
    pel_dx = float(trajectories[-1]["actual_positions_m"][0][0] - trajectories[0]["actual_positions_m"][0][0])
    la_max_z = float(max(r["actual_positions_m"][1][2] for r in trajectories))
    ra_max_z = float(max(r["actual_positions_m"][2][2] for r in trajectories))
    la_dx = float(trajectories[-1]["actual_positions_m"][1][0] - trajectories[0]["actual_positions_m"][1][0])
    ra_dx = float(trajectories[-1]["actual_positions_m"][2][0] - trajectories[0]["actual_positions_m"][2][0])
    return {
        "video": video_path.name,
        "scenario": scenario,
        "seed": seed,
        "mode": mode_label,
        "robot": robot,
        "bridge": bridge,
        "success": reason == "success",
        "termination": reason,
        "steps": k + 1,
        "min_wrist_distance_m": closest,
        "hand_contact_steps": contact_steps,
        "button_max_travel_m": None,
        "pelvis_net_dx_m": round(pel_dx, 4),
        "left_ankle_net_dx_m": round(la_dx, 4),
        "right_ankle_net_dx_m": round(ra_dx, 4),
        "ankle_max_z_m": [round(la_max_z, 4), round(ra_max_z, 4)],
        "planning_calls": updates,
        "physics": env.physics_diagnostics.report(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--wam-run", required=True, type=Path)
    parser.add_argument("--wam-checkpoint", required=True, type=Path)
    parser.add_argument("--mimic12k-run", required=True, type=Path)
    parser.add_argument("--mimic40k-run", required=True, type=Path)
    parser.add_argument("--gripper-processed", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda:0"
    config = json.loads((args.wam_run / "run.json").read_text())
    saved = torch.load(args.wam_checkpoint, map_location="cpu", weights_only=True)
    dit_block_ids = {int(k.split(".")[2]) for k in saved["expert"] if k.startswith("backbone.transformer_blocks.")}
    backbone_path = Path("/data/kk/models/nvidia/Cosmos-Predict2.5-2B")
    backbone_manifest = Path("/data/kk/data/wammimic_downloads_20261116/cosmos_backbone_20261118.json")
    wam = CosmosWAM(
        load_backbone(backbone_path, backbone_manifest, device),
        width=config["width"],
        layers=config["layers"],
        unfreeze_blocks=len(dit_block_ids),
        use_proprio=True,
        num_embodiments=config["num_embodiments"],
    ).to(device).eval()
    wam.load_state_dict(saved["expert"], strict=False)
    vae = VideoEncoder("/data/kk/models/wammimic_cosmos25_encoders_20261118")

    # Load exact OpenHLM task_0 prompt embedding & stats
    with safe_open(str(args.gripper_processed / "prompt_bank_bf16.safetensors"), framework="pt", device="cpu") as f:
        text_t0 = f.get_tensor("task_0")[None].to(device)
    gripper_stats = json.loads((args.gripper_processed / "wam_lerobot_v2/g1_parallel_gripper/meta/stats.json").read_text())
    act_stats = gripper_stats["action"]
    st_stats = gripper_stats["observation.state_5pt_hist"]

    # Define 4 controlled evaluations on the exact OpenHLM task_0 red-can tabletop scene
    configs = [
        {
            "scenario": "openhlm-red-can-q1",
            "mode_label": "wam_mimic",
            "robot": "q1",
            "scale": SCALE_Q1,
            "spawn": np.array([-0.43, 0.0, 0.0]),
            "fovy": 72.0,
            "bridge": "raw",
            "mimic_run": args.mimic12k_run,
            "mimic_step": 12000,
        },
        {
            "scenario": "openhlm-red-can-q1",
            "mode_label": "wam_mimic_yaw_40k",
            "robot": "q1",
            "scale": SCALE_Q1,
            "spawn": np.array([-0.43, 0.0, 0.0]),
            "fovy": 72.0,
            "bridge": "yaw_retarget",
            "mimic_run": args.mimic40k_run,
            "mimic_step": 40000,
        },
        {
            "scenario": "openhlm-red-can-g1",
            "mode_label": "wam_mimic_g1_raw_40k",
            "robot": "g1",
            "scale": 1.0,
            "spawn": np.array([-0.75, 0.0, 0.0]),
            "fovy": 92.0,
            "bridge": "raw",
            "mimic_run": args.mimic40k_run,
            "mimic_step": 40000,
        },
        {
            "scenario": "openhlm-red-can-g1",
            "mode_label": "wam_mimic_g1_yaw_40k",
            "robot": "g1",
            "scale": 1.0,
            "spawn": np.array([-0.75, 0.0, 0.0]),
            "fovy": 92.0,
            "bridge": "yaw_retarget",
            "mimic_run": args.mimic40k_run,
            "mimic_step": 40000,
        },
    ]

    ref_dir = args.out_dir / "reference"
    cases = []
    for spec_cfg in configs:
        robot = spec_cfg["robot"]
        mrun = spec_cfg["mimic_run"]
        mstep = spec_cfg["mimic_step"]
        mcfg = json.loads((mrun / "config.json").read_text())
        mdata = Path(mcfg["prepared_manifest"]).parent
        ckpt = mrun / robot / f"ckpt_{mstep}.pt"
        policy, meta = load_policy(
            ckpt, "cpu", interface="human", human_points="five",
            kp_horizon=GOAL_K, goal_anchor=mcfg["goal_anchor"],
        )
        subset_yaml, rests = prepare_reference(ref_dir, mdata, robot, spec_cfg["spawn"], args.steps + 20)
        cfg = morphmimic_env_cfg(
            robot, subset_yaml=str(subset_yaml), num_envs=1, play=True, eval=True,
            goal_anchor=mcfg["goal_anchor"], ref_feedback=True,
        )
        cfg.scene.spec_fn = build_red_can_scene(
            args.run / "assets", scale=spec_cfg["scale"], robot=robot, fovy=spec_cfg["fovy"],
        )
        env = CpuPreviewEnv(cfg, device="cpu")
        try:
            for seed in args.seeds:
                vname = f"{spec_cfg['scenario']}-{spec_cfg['mode_label']}-seed{seed}.mp4"
                res = run_episode(
                    env, policy, meta, wam, vae, text_t0, act_stats, st_stats, rests,
                    robot=robot, bridge=spec_cfg["bridge"],
                    scenario=spec_cfg["scenario"], mode_label=spec_cfg["mode_label"],
                    seed=seed, video_path=args.out_dir / vname, steps=args.steps, emb_id=1,
                )
                cases.append(res)
                summary = {k: v for k, v in res.items() if k not in ("planning_calls", "physics")}
                print(json.dumps(summary), flush=True)
        finally:
            env.close()

    write_json(args.out_dir / "manifest.json", {
        "kind": "geniesim_q1",
        "step": 5000,
        "mimic_step": "12000 / 40000",
        "wam_checkpoint_sha256": sha256(args.wam_checkpoint),
        "backend": "MuJoCo CPU",
        "robot": "Q1 & G1 (OpenHLM Task 0 同分布对照)",
        "prompt": "Walk forward to the table, put the red can on the orange mouse pad with the right hand, turn around and walk back.",
        "cases": cases,
    })


if __name__ == "__main__":
    main()
