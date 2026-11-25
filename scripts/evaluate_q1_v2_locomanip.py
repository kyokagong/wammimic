"""Evaluate Canonical Human v2.0 WAM + Q1 Mimic 40k on loco-manipulation tasks in Genie Sim.

Uses the exact bidirectional Canonical Human v2.0 bridge (wammimic.data_pipeline.canonical_human)
and global normalization statistics (canonical_human_5pt_stats.json) on the Q1 robot body across:
  1. openhlm-red-can-q1: OpenHLM task_0 in-distribution tabletop loco-manipulation
  2. press-button: Genie Sim tabletop approach & button press
  3. matching-block-contact: Genie Sim tabletop approach & block contact
"""
from __future__ import annotations

import argparse
from collections import deque
from contextlib import ExitStack
import json
from pathlib import Path
import subprocess
from time import perf_counter

import mujoco
import numpy as np
from safetensors import safe_open
from safetensors.torch import load_file
from scipy.spatial.transform import Rotation, Slerp
import torch

from morphmimic.constants import GOAL_K, PROPRIO_DIM
from morphmimic.env.env_cfg import COMMAND_NAME, morphmimic_env_cfg
from morphmimic.eval.policy_loader import load_policy
from morphmimic.model.state import split_state
from morphmimic.train.obs_history import ObsHistory
from preview.cpu_backend import CpuPreviewEnv
from scripts.evaluate_geniesim_indist import SCALE_Q1, build_red_can_scene, prepare_reference
from scripts.evaluate_geniesim_wam_mimic import (
    DT,
    SLOTS,
    TASKS,
    actor_targets,
    interpolate,
    predict,
    render_panel,
    scene_callback,
)
from wammimic.data_pipeline.cache import VideoEncoder
from wammimic.data_pipeline.canonical_human import (
    ROBOT_GEOMETRY,
    canonical_5pt_to_target_world,
    world_flu_to_canonical_5pt,
    world_flu_to_canonical_history,
)
from wammimic.data_pipeline.common import sha256, write_json
from wammimic.data_pipeline.geometry import matrices
from wammimic.training.model import CosmosWAM, load_backbone


def _blend_world_chunks(old_chunk, new_chunk, old_offset: int, max_blend_frames: int = 25):
    """Smoothly blend remaining old 50Hz trajectory into new 50Hz trajectory via raised-cosine + SLERP."""
    old_pos, old_rot = old_chunk
    new_pos, new_rot = new_chunk
    rem = len(old_pos) - old_offset
    blend_n = min(rem, len(new_pos), max_blend_frames)
    if blend_n <= 1:
        return new_pos, new_rot
    out_pos = new_pos.copy()
    out_rot = new_rot.copy()
    # w goes from 1.0 (old) at j=0 down to 0.0 (new) at j=blend_n-1
    u = np.linspace(0.0, 1.0, blend_n, dtype=np.float64)
    w = 0.5 * (1.0 + np.cos(np.pi * u))
    out_pos[:blend_n] = w[:, None, None] * old_pos[old_offset : old_offset + blend_n] + (1.0 - w)[:, None, None] * new_pos[:blend_n]
    for j in range(blend_n):
        wj = float(w[j])
        if wj >= 0.999:
            out_rot[j] = old_rot[old_offset + j]
        elif wj <= 0.001:
            out_rot[j] = new_rot[j]
        else:
            for p_i in range(5):
                slerp = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([new_rot[j, p_i], old_rot[old_offset + j, p_i]])))
                out_rot[j, p_i] = slerp([wj]).as_matrix()[0]
    return out_pos, out_rot


def _rot_to_wxyz(rot: np.ndarray) -> np.ndarray:
    xyzw = Rotation.from_matrix(rot.reshape(-1, 3, 3)).as_quat()
    return xyzw[:, [3, 0, 1, 2]].reshape(rot.shape[:-2] + (4,))


def canonical_v2_q1_targets(
    pred_45: np.ndarray,
    actual_5pt: np.ndarray,
    rests: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    anchor_xy: np.ndarray | None = None,
    max_lead_m: float = 0.08,
):
    """Map denormalized Canonical Human v2.0 [10, 45] prediction to Q1 Mimic human & robot targets."""
    robot_rest_pos, robot_rest_rot, human_rest_pos, human_rest_rot = rests
    geom = ROBOT_GEOMETRY["q1"]

    anchor = actual_5pt.copy().astype(np.float64)
    origin_xy = anchor[0, :2].copy()
    if anchor_xy is not None:
        lead = anchor_xy - origin_xy
        norm = float(np.linalg.norm(lead))
        if norm > max_lead_m:
            lead = lead * (max_lead_m / norm)
        origin_xy = origin_xy + lead
    anchor[0, 0] = origin_xy[0]
    anchor[0, 1] = origin_xy[1]
    # Anchor pelvis Z to standing height so minor crouch/tracking lag never drifts downward.
    anchor[0, 2] = float(robot_rest_pos[0, 2])

    target_world_5pt = canonical_5pt_to_target_world(
        pred_45,
        anchor,
        geom["leg_height"],
        geom["arm_length"],
        ankle_min_z=geom["ankle_rest_z"] - 0.005,
    )
    goal = target_world_5pt[:, :, :3].astype(np.float64)
    goal_rot = matrices(target_world_5pt[:, :, 3:])

    spawn_xy = np.array([robot_rest_pos[0, 0], robot_rest_pos[0, 1], 0.0], dtype=np.float64)
    human = (human_rest_pos + spawn_xy)[None] + (goal - robot_rest_pos[None]) * 1.65
    human_rot = goal_rot @ robot_rest_rot.swapaxes(-1, -2) @ human_rest_rot
    return (human, human_rot), (goal, goal_rot)


@torch.inference_mode()
def run_q1_episode(
    env,
    policy,
    meta,
    wam,
    vae,
    text,
    gstats: dict,
    rests: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    scenario: str,
    mode: str,
    seed: int,
    video_path: Path,
    steps: int,
    emb_id: int = 1,
    replan_steps: int = 10,
    use_rtc: bool = False,
):
    geom = ROBOT_GEOMETRY["q1"]
    state_ids = torch.arange(1, device=env.device)
    command = env.command_manager.get_term(COMMAND_NAME)
    command.assign_clips(state_ids, torch.zeros_like(state_ids))
    obs, _ = env.reset(env_ids=state_ids)
    history = ObsHistory(1, meta.hist_len, PROPRIO_DIM, env.device)
    history.reset(state_ids, obs["policy"])
    generator = torch.Generator(device="cuda:0").manual_seed(seed) if mode != "hold" else None

    model = env.sim.mj_model
    data = env.native_data
    model.vis.global_.offwidth, model.vis.global_.offheight = 960, 540
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [float(rests[0][0, 0]) + 0.55, float(rests[0][0, 1]), 0.36]
    camera.distance, camera.azimuth, camera.elevation = 2.40, 128.0, -18.0

    target = model.geom("task_target").id
    hand_geoms = {
        g for g in range(model.ngeom)
        if "robot/" in model.body(model.geom_bodyid[g]).name
        and any(term in model.body(model.geom_bodyid[g]).name for term in ("wrist", "hand"))
    }
    if not hand_geoms:
        raise ValueError("Q1 scene has no hand contact geometry")
    button_qadr = None if scenario != "press-button" else int(model.jnt_qposadr[model.joint("button_travel").id])

    act_mean = np.asarray(gstats["action"]["mean"], dtype=np.float32)
    act_std = np.maximum(np.asarray(gstats["action"]["std"], dtype=np.float32), 1e-6)
    prop_mean = np.asarray(gstats["proprio_5pt"]["mean"], dtype=np.float32)
    prop_std = np.maximum(np.asarray(gstats["proprio_5pt"]["std"], dtype=np.float32), 1e-6)

    frames = deque(maxlen=4)
    updates, contact_steps = [], 0
    closest, depth, reason = np.inf, 0.0, "timeout"
    human = robot_tgt = None
    spawn_xy = np.array([rests[0][0, 0], rests[0][0, 1], 0.0], dtype=np.float64)
    human_start = rests[2] + spawn_xy
    trajectories = []
    poses_50hz = []

    mode_title = "hold baseline" if mode == "hold" else "WAM v2.0 + Q1 Mimic 40k"
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
            actual_5pt = np.concatenate((actual_pos, actual_quat), axis=-1)
            poses_50hz.append(actual_5pt)

            if k % 5 == 0:
                head.update_scene(data, camera="wam_head")
                last_image = head.render().copy()
                frames.append(last_image)

            if mode != "hold" and len(frames) == 4 and (k - 15) % replan_steps == 0:
                started = perf_counter()
                old = k - updates[-1]["control_step"] if updates else 0
                anchor_xy = None if robot_tgt is None else robot_tgt[0][old, 0, :2]

                p_arr = np.asarray(poses_50hz, dtype=np.float64)
                t_arr = np.arange(len(p_arr), dtype=np.float64) * DT
                hist_idx = np.array([[k - 15, k - 10, k - 5, k]], dtype=np.int64)
                proprio_raw = world_flu_to_canonical_history(
                    p_arr, t_arr, hist_idx, geom["leg_height"], geom["arm_length"], allow_zero_initial=True,
                )[0]
                proprio_norm = ((proprio_raw - prop_mean) / prop_std).astype(np.float32)
                ee_h4 = np.zeros((4, 4), dtype=np.float32)

                prefix_target = None
                prefix_weights = None
                if use_rtc and robot_tgt is not None and old < 50:
                    rem_indices = list(range(old + 5, 51, 5))
                    n_rem = len(rem_indices)
                    if n_rem > 0:
                        rem_pos = robot_tgt[0][rem_indices]
                        rem_quat = _rot_to_wxyz(robot_tgt[1][rem_indices])
                        rem_world_5pt = np.concatenate([rem_pos, rem_quat], axis=-1)
                        # Project old remaining steps into the same anchor frame used by canonical_v2_q1_targets
                        anchor_rtc = actual_5pt.copy().astype(np.float64)
                        orig_xy = anchor_rtc[0, :2].copy()
                        if anchor_xy is not None:
                            lead = anchor_xy - orig_xy
                            nrm = float(np.linalg.norm(lead))
                            if nrm > 0.08:
                                lead = lead * (0.08 / nrm)
                            orig_xy = orig_xy + lead
                        anchor_rtc[0, 0] = orig_xy[0]
                        anchor_rtc[0, 1] = orig_xy[1]
                        anchor_rtc[0, 2] = float(rests[0][0, 2])
                        rem_canon_45 = world_flu_to_canonical_5pt(
                            rem_world_5pt, anchor_rtc, geom["leg_height"], geom["arm_length"],
                        )
                        prefix_target = np.zeros((10, 47), dtype=np.float32)
                        prefix_target[:n_rem, :45] = ((rem_canon_45 - act_mean[:45]) / act_std[:45]).astype(np.float32)
                        prefix_weights = np.zeros((10,), dtype=np.float32)
                        for j in range(n_rem):
                            prefix_weights[j] = float(0.5 * (1.0 + np.cos(np.pi * (j + 0.5) / n_rem)))

                pred_denorm = predict(
                    wam, vae, list(frames), text, generator, act_mean, act_std,
                    proprio=proprio_norm, ee_hist=ee_h4, embodiment_id=emb_id,
                    prefix_target=prefix_target, prefix_weights=prefix_weights,
                )
                human_block, robot_block = canonical_v2_q1_targets(
                    pred_denorm[:, :45], actual_5pt, rests, anchor_xy=anchor_xy,
                )
                start_h = (human_start, rests[3]) if human is None else (human[0][old], human[1][old])
                start_r = (actual_pos, actual_rot) if robot_tgt is None else (robot_tgt[0][old], robot_tgt[1][old])
                new_human, new_robot = interpolate(start_h, human_block), interpolate(start_r, robot_block)
                if use_rtc and human is not None and robot_tgt is not None:
                    human = _blend_world_chunks(human, new_human, old, max_blend_frames=25)
                    robot_tgt = _blend_world_chunks(robot_tgt, new_robot, old, max_blend_frames=25)
                else:
                    human, robot_tgt = new_human, new_robot
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
            if button_qadr is not None and contact:
                depth = max(depth, float(data.qpos[button_qadr]))

            action = policy(history.flat(), targets)
            if not bool(torch.isfinite(action).all()):
                raise ValueError("Mimic produced nonfinite action")

            trajectories.append({
                "step": k,
                "wrist_target_distance_m": distance,
                "hand_contact": contact,
                "button_travel_m": None if button_qadr is None else float(data.qpos[button_qadr]),
                "task_object_position_m": data.geom_xpos[target].tolist(),
                "target_positions_m": None if markers is None else markers.tolist(),
                "actual_positions_m": actual_pos.tolist(),
            })
            reached = depth >= 0.003 if button_qadr is not None else contact_steps > 0
            if k % 2 == 0 or reached:
                dx_now = float(actual_pos[0, 0] - rests[0][0, 0])
                camera.lookat[:] = [float(actual_pos[0, 0]) + 0.55, float(actual_pos[0, 1]), 0.36]
                label = f"{scenario} / {mode_title} / seed {seed}   {k * DT:.2f}s   dx {dx_now:+.2f}m   closest {closest*100:.1f}cm"
                if button_qadr is not None:
                    label += f"   travel {depth*1000:.1f}mm"
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

    diagnostics = env.physics_diagnostics.report()
    if diagnostics["overflow_bits"]:
        reason = "physics_overflow"
    video_path.with_suffix(".jsonl").write_text("".join(json.dumps(r) + "\n" for r in trajectories))

    pel_dx = float(trajectories[-1]["actual_positions_m"][0][0] - trajectories[0]["actual_positions_m"][0][0])
    la_max_z = float(max(r["actual_positions_m"][1][2] for r in trajectories))
    ra_max_z = float(max(r["actual_positions_m"][2][2] for r in trajectories))
    return {
        "video": video_path.name,
        "scenario": scenario,
        "seed": seed,
        "mode": mode,
        "robot": "q1",
        "bridge": "canonical_human_v2",
        "success": reason == "success",
        "termination": reason,
        "steps": k + 1,
        "min_wrist_distance_m": closest,
        "hand_contact_steps": contact_steps,
        "button_max_travel_m": None if button_qadr is None else depth,
        "pelvis_net_dx_m": round(pel_dx, 4),
        "ankle_max_z_m": [round(la_max_z, 4), round(ra_max_z, 4)],
        "planning_calls": updates,
        "physics": diagnostics,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="Genie Sim workspace root (with assets, genie_sim, text)")
    parser.add_argument("--wam-run", required=True, type=Path)
    parser.add_argument("--wam-checkpoint", required=True, type=Path)
    parser.add_argument("--mimic-run", required=True, type=Path)
    parser.add_argument("--mimic-step", type=int, default=40000)
    parser.add_argument("--p1-processed", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=750)
    parser.add_argument("--spawn-x", type=float, default=-3.2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--embodiment-id", type=int, default=1)
    parser.add_argument("--replan-steps", type=int, default=10)
    parser.add_argument("--only-openhlm", action="store_true")
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

    gstats = json.loads((args.p1_processed / "canonical_human_5pt_stats.json").read_text())

    # Load Q1 Mimic policy
    mcfg = json.loads((args.mimic_run / "config.json").read_text())
    if not mcfg.get("ref_feedback", False):
        raise ValueError("Missing Global + Ref. interface in Mimic run")
    mdata = Path(mcfg["prepared_manifest"]).parent
    mimic_ckpt = args.mimic_run / "q1" / f"ckpt_{args.mimic_step}.pt"
    policy, meta = load_policy(
        mimic_ckpt, "cpu", interface="human", human_points="five",
        kp_horizon=GOAL_K, goal_anchor=mcfg["goal_anchor"],
    )
    assert meta.robot == "q1" and meta.iteration == args.mimic_step

    # Prompt embeddings: OpenHLM task_0 + GenieSim task_0/task_1
    with safe_open(str(args.p1_processed / "gripper/prompt_bank_bf16.safetensors"), framework="pt", device="cpu") as f:
        openhlm_t0 = f.get_tensor("task_0")[None].to(device)
    genie_bank = load_file(args.run / "text/prompt_bank_bf16.safetensors", device=device)

    ref_dir = args.out_dir / "reference"
    spawn_q1 = np.array([args.spawn_x, 0.0, 0.0], dtype=np.float64)
    subset_yaml, rests = prepare_reference(ref_dir, mdata, "q1", spawn_q1, args.steps + 50)

    source_tasks = args.run / "genie_sim/source/geniesim_scene_agent/examples/tasks"
    scenarios = [
        ("openhlm-red-can-q1", openhlm_t0, build_red_can_scene(args.run / "assets", scale=SCALE_Q1, robot="q1", fovy=72.0)),
    ]
    if not args.only_openhlm:
        for idx, (scenario, task, layout, target) in enumerate(TASKS):
            scene_path = source_tasks / task / layout / "scene_info.json"
            scene = json.loads(scene_path.read_text())
            scenarios.append((scenario, genie_bank[f"task_{idx}"][None], scene_callback(scene, args.run / "assets", target)))

    cases = []
    for scenario, text_emb, spec_fn in scenarios:
        cfg = morphmimic_env_cfg(
            "q1", subset_yaml=str(subset_yaml), num_envs=1, play=True, eval=True,
            goal_anchor=mcfg["goal_anchor"], ref_feedback=True,
        )
        cfg.scene.spec_fn = spec_fn
        env = CpuPreviewEnv(cfg, device="cpu")
        try:
            jobs = [("hold", args.seeds[0]), *(("wam_mimic_v2", s) for s in args.seeds)]
            for mode, seed in jobs:
                vname = f"{scenario}-{f'seed{seed}' if mode != 'hold' else 'hold'}.mp4"
                res = run_q1_episode(
                    env, policy, meta, wam, vae, text_emb, gstats, rests,
                    scenario=scenario, mode=mode, seed=seed,
                    video_path=args.out_dir / vname, steps=args.steps,
                    emb_id=args.embodiment_id, replan_steps=args.replan_steps,
                )
                cases.append(res)
                summary = {k: v for k, v in res.items() if k not in ("planning_calls", "physics")}
                print(json.dumps(summary), flush=True)
        finally:
            env.close()

    manifest_path = args.out_dir / "manifest.json"
    preserved_real = []
    if manifest_path.is_file():
        old_m = json.loads(manifest_path.read_text())
        preserved_real = [c for c in old_m.get("cases", []) if c.get("mode") == "real_sample"]

    write_json(manifest_path, {
        "kind": "geniesim_q1",
        "step": int(saved["step"]),
        "mimic_step": int(meta.iteration),
        "wam_checkpoint_sha256": sha256(args.wam_checkpoint),
        "mimic_checkpoint_sha256": sha256(mimic_ckpt),
        "backend": "MuJoCo CPU",
        "robot": f"Q1 (Canonical Human v2.0 + Mimic 40k · spawn {abs(args.spawn_x):.1f}m · {args.steps*DT:.0f}s)",
        "cases": cases + preserved_real,
    })


if __name__ == "__main__":
    main()
