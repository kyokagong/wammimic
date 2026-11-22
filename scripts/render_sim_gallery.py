"""Render 3rd-person + inset head-camera MP4 videos and manifest.json for collected D1 sim episodes."""
import argparse
import json
import os
import subprocess
from contextlib import ExitStack
from pathlib import Path

import mujoco
import numpy as np

from scripts.evaluate_geniesim_wam_mimic import (
    DT,
    CpuPreviewEnv,
    morphmimic_env_cfg,
    render_panel,
    scene_callback,
)

TASKS = {
    "press_button": "benchmark_button_1dbfb257",
    "avoid_obstacle_then_press_button": "benchmark_button_1dbfb257",
    "matching_block_contact": "benchmark_building_blocks_d5d636e1",
}


def render_episode(ep_dir: Path, out_mp4: Path, assets: Path, subset_yaml: Path, goal_anchor: str) -> dict:
    ep = json.loads((ep_dir / "episode.json").read_text())
    traj = np.load(ep_dir / "trajectory.npz")
    poses = traj["poses_world"]          # (T, 5, 7) [x,y,z, qw,qx,qy,qz]
    joint_pos = traj["joint_pos"]        # (T, n_joints)
    rgb = traj["rgb"]                    # (N_rgb, 224, 224, 3)
    rgb_idx = traj["rgb_frame_index"]    # (N_rgb,)
    touch = traj["touch"]                # (T, 2)
    task = ep.get("task", "press_button")
    target_name = TASKS.get(task, "button_001")
    obs_meta = ep.get("obstacle")
    base = scene_callback(ep["scene"], assets / "assets", target_name)

    def configure(spec):
        base(spec)
        if obs_meta is not None:
            body = spec.worldbody.add_body(name="floor_obstacle_body", pos=obs_meta["pos"])
            body.add_geom(
                name="floor_obstacle",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=obs_meta["half_size"],
                rgba=[0.95, 0.38, 0.08, 1.0],
                contype=1,
                conaffinity=1,
            )

    cfg = morphmimic_env_cfg(
        "q1", subset_yaml=str(subset_yaml), num_envs=1,
        play=True, eval=True, goal_anchor=goal_anchor, ref_feedback=True,
    )
    cfg.scene.spec_fn = configure
    env = CpuPreviewEnv(cfg, device="cpu")
    try:
        model = env.sim.mj_model
        data = env.native_data
        model.vis.global_.offwidth, model.vis.global_.offheight = 960, 540
        target = model.geom("task_target").id
        mujoco.mj_forward(model, data)
        target_xy = data.geom_xpos[target, :2].copy()
        detour_side = int(obs_meta.get("detour_side", 1)) if obs_meta else 1

        camera = mujoco.MjvCamera()
        # Wide elevated 3/4 view framing spawn, floor obstacle, curved detour path, table, and button.
        camera.lookat[:] = [float(0.5 * (-0.43 + target_xy[0])), float(0.45 * target_xy[1]), 0.34]
        camera.distance = 2.15
        camera.azimuth = 132.0 if detour_side >= 0 else 228.0
        camera.elevation = -26.0

        button_qadr = None
        try:
            button_qadr = int(model.jnt_qposadr[model.joint("button_travel").id])
        except Exception:
            pass
        max_travel = float(ep.get("button_max_travel_m") or 0.0)
        contact_frames = np.where(touch.max(axis=-1) > 0.5)[0]
        first_contact = int(contact_frames[0]) if len(contact_frames) else len(poses)

        T = len(poses)
        with ExitStack() as stack:
            wide = stack.enter_context(mujoco.Renderer(model, height=540, width=960))
            video = stack.enter_context(subprocess.Popen(
                ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                 "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "960x540", "-r", "25",
                 "-i", "pipe:0", "-c:v", "libx264", "-threads", "2", "-pix_fmt", "yuv420p", str(out_mp4)],
                stdin=subprocess.PIPE,
            ))
            for k in range(0, T, 2):
                data.qpos[0:3] = poses[k, 0, 0:3]
                data.qpos[3:7] = poses[k, 0, 3:7]
                data.qpos[env.joint_qadr] = joint_pos[k]
                if button_qadr is not None and k >= first_contact:
                    data.qpos[button_qadr] = max_travel
                elif button_qadr is not None:
                    data.qpos[button_qadr] = 0.0
                mujoco.mj_forward(model, data)
                r_i = int(np.searchsorted(rgb_idx, k, side="right")) - 1
                r_i = max(0, min(r_i, len(rgb) - 1))
                side_lbl = "L-detour" if detour_side > 0 else "R-detour"
                pressed = "PRESSED" if k >= first_contact else "approaching"
                status = (
                    f"seed {ep['seed']}  |  {side_lbl}  |  t={k * DT:.2f}s  |  "
                    f"detour {ep.get('max_lateral_detour_m', 0.0):.2f}m  |  "
                    f"obs_col {ep.get('obstacle_collision_steps', 0)}  |  {pressed}"
                )
                frame = render_panel(wide, data, camera, rgb[r_i], poses[k, :, :3], status)
                video.stdin.write(frame.tobytes())
            # Hold final frame for 0.6s so button press is easy to inspect.
            video.stdin.write(frame.tobytes() * 15)
            video.stdin.close()
            if video.wait() != 0:
                raise RuntimeError(f"ffmpeg failed for {out_mp4}")
    finally:
        env.close()

    return {
        "seed": ep["seed"],
        "task": task,
        "accepted": bool(ep.get("accepted", True)),
        "instruction": ep.get("instruction", ""),
        "video": out_mp4.name,
        "max_lateral_detour_m": ep.get("max_lateral_detour_m", 0.0),
        "obstacle_collision_steps": ep.get("obstacle_collision_steps", 0),
        "button_max_travel_m": ep.get("button_max_travel_m", 0.0),
        "target_in_frustum_fraction": ep.get("target_in_frustum_fraction", 0.0),
        "target_and_wrist_in_frustum_fraction": ep.get("target_and_wrist_in_frustum_fraction", 0.0),
        "detour_side": obs_meta.get("detour_side") if obs_meta else 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--task", type=str, default="avoid_obstacle_then_press_button")
    parser.add_argument("--max-episodes", type=int, default=12)
    args = parser.parse_args()

    assets = Path(os.environ["SIM_ASSET_ROOT"])
    mimic = Path(os.environ["MIMIC_RUN_ROOT"])
    raw_sim = Path(os.environ["RAW_DATA_ROOT"]) / "sim_loco_tabletop"
    frozen = json.loads((mimic / "config.json").read_text())
    subset_yaml = raw_sim / "reference/subset.yaml"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    candidates = []
    for ep_dir in sorted(args.episodes_dir.glob("seed_*")):
        ep_json = ep_dir / "episode.json"
        if not ep_json.is_file():
            continue
        meta = json.loads(ep_json.read_text())
        if args.task and meta.get("task") != args.task:
            continue
        candidates.append((ep_dir, meta))
    candidates = candidates[:args.max_episodes]
    if not candidates:
        raise SystemExit(f"No episodes matching task={args.task!r} in {args.episodes_dir}")

    entries = []
    for ep_dir, meta in candidates:
        out_mp4 = args.out_dir / f"{ep_dir.name}.mp4"
        print(f"Rendering {ep_dir.name} -> {out_mp4.name} ...", flush=True)
        entry = render_episode(ep_dir, out_mp4, assets, subset_yaml, frozen["goal_anchor"])
        entries.append(entry)

    manifest = {"task": args.task, "count": len(entries), "episodes": entries}
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"Wrote {len(entries)} videos and manifest.json to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
