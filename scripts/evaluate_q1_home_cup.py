"""Evaluate WAM v2.0 + Q1 Mimic 40k in a realistic indoor home room walking to a table and touching a water cup."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import mujoco
import numpy as np
from safetensors import safe_open
import torch

from morphmimic.constants import GOAL_K
from morphmimic.env.env_cfg import morphmimic_env_cfg
from morphmimic.eval.policy_loader import load_policy
from preview.cpu_backend import CpuPreviewEnv
from scripts.evaluate_geniesim_indist import SCALE_Q1, prepare_reference
from scripts.evaluate_geniesim_wam_mimic import DT, usd_geometry
from scripts.evaluate_q1_v2_locomanip import run_q1_episode
from wammimic.data_pipeline.cache import VideoEncoder
from wammimic.data_pipeline.common import sha256, write_json
from wammimic.training.model import CosmosWAM, load_backbone


def build_home_cup_scene(assets: Path, *, scale: float = SCALE_Q1, fovy: float = 86.0):
    """Build a realistic textured indoor home room with hardwood floor, walls, furniture, table, blue cup & white plate."""
    table_meshes = usd_geometry(assets / "objects/edited/table_000/Aligned.usda")
    table_xyz = np.array([0.0, 0.0, 0.0], dtype=float)
    source_top = table_xyz[2] + max(pts[:, 2].max() for _, pts, _ in table_meshes)
    top = source_top * scale

    cup_radius = 0.036 * scale
    cup_half_h = 0.055 * scale
    plate_radius = 0.095 * scale
    plate_half_h = 0.008 * scale

    def configure(spec):
        # Level wide-angle ego camera (8 deg slight pitch) so table + blue cup stay framed from 3.2m away
        torso = spec.body("robot/torso_link")
        pitch = np.deg2rad(8.0)
        torso.add_camera(
            name="wam_head",
            pos=[0.085, 0.0, 0.23],
            fovy=fovy,
            xyaxes=[0.0, -1.0, 0.0, np.sin(pitch), 0.0, np.cos(pitch)],
        )

        # Warm hardwood floor material + matte taupe area rug (no glare/overexposure)
        wood_tex = spec.add_texture(
            name="home_wood_tex",
            type=mujoco.mjtTexture.mjTEXTURE_2D,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
            rgb1=[0.46, 0.34, 0.22],
            rgb2=[0.38, 0.27, 0.17],
            width=256,
            height=256,
        )
        wood_mat = spec.add_material(name="home_wood_mat", texrepeat=[14.0, 14.0], reflectance=0.02)
        wood_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "home_wood_tex"

        wall_tex = spec.add_texture(
            name="home_wall_tex",
            type=mujoco.mjtTexture.mjTEXTURE_2D,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
            rgb1=[0.78, 0.76, 0.71],
            rgb2=[0.66, 0.63, 0.58],
            width=128,
            height=128,
        )
        wall_mat = spec.add_material(name="home_wall_mat", texrepeat=[1.0, 1.0], reflectance=0.0)
        wall_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "home_wall_tex"

        rug_tex = spec.add_texture(
            name="home_rug_tex",
            type=mujoco.mjtTexture.mjTEXTURE_2D,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
            rgb1=[0.42, 0.40, 0.37],
            rgb2=[0.35, 0.33, 0.30],
            width=128,
            height=128,
        )
        rug_mat = spec.add_material(name="home_rug_mat", texrepeat=[8.0, 6.0], reflectance=0.0)
        rug_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "home_rug_tex"

        # Visual wood floor overlay (non-colliding, sits 1mm above physics floor)
        spec.worldbody.add_geom(
            name="home_wood_floor",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[-1.6, 0.0, -0.004],
            size=[4.2, 3.2, 0.005],
            material="home_wood_mat",
            contype=0,
            conaffinity=0,
        )
        # Woven living-room rug leading up to the dining table
        spec.worldbody.add_geom(
            name="home_living_rug",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[-1.5, 0.0, 0.0005],
            size=[1.4, 1.05, 0.001],
            material="home_rug_mat",
            contype=0,
            conaffinity=0,
        )

        # Room walls (enclosed home interior: front wall behind table, left/right walls, back wall)
        room_walls = [
            ("wall_front", [1.35, 0.0, 1.35], [0.06, 3.1, 1.35]),
            ("wall_back", [-4.6, 0.0, 1.35], [0.06, 3.1, 1.35]),
            ("wall_left", [-1.6, 2.6, 1.35], [3.1, 0.06, 1.35]),
            ("wall_right", [-1.6, -2.6, 1.35], [3.1, 0.06, 1.35]),
        ]
        for wname, wpos, wsize in room_walls:
            spec.worldbody.add_geom(
                name=wname,
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=wpos,
                size=wsize,
                material="home_wall_mat",
                contype=0,
                conaffinity=0,
            )

        # Wooden baseboards for clear horizon/perspective cues
        for bname, bpos, bsize in [
            ("baseboard_front", [1.28, 0.0, 0.06], [0.015, 3.0, 0.06]),
            ("baseboard_left", [-1.6, 2.53, 0.06], [3.0, 0.015, 0.06]),
            ("baseboard_right", [-1.6, -2.53, 0.06], [3.0, 0.015, 0.06]),
        ]:
            spec.worldbody.add_geom(
                name=bname, type=mujoco.mjtGeom.mjGEOM_BOX, pos=bpos, size=bsize,
                rgba=(0.36, 0.25, 0.16, 1.0), contype=0, conaffinity=0,
            )

        # Framed window on front wall + wooden credenza/bookshelf on left & sofa on right for realistic home context
        spec.worldbody.add_geom(
            name="window_glass", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[1.27, 0.0, 1.30], size=[0.015, 0.95, 0.55],
            rgba=(0.76, 0.88, 0.98, 1.0), contype=0, conaffinity=0,
        )
        spec.worldbody.add_geom(
            name="window_frame_bar", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[1.255, 0.0, 1.30], size=[0.02, 0.025, 0.57],
            rgba=(0.95, 0.95, 0.93, 1.0), contype=0, conaffinity=0,
        )
        # Sideboard / cabinet on left wall
        spec.worldbody.add_geom(
            name="home_cabinet", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[-1.2, 2.15, 0.40], size=[0.85, 0.28, 0.40],
            rgba=(0.45, 0.31, 0.20, 1.0), contype=0, conaffinity=0,
        )
        # Potted plant next to table
        spec.worldbody.add_geom(
            name="plant_pot", type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=[0.45, 1.15, 0.18], size=[0.14, 0.18, 0.0],
            rgba=(0.72, 0.42, 0.26, 1.0), contype=0, conaffinity=0,
        )
        spec.worldbody.add_geom(
            name="plant_foliage", type=mujoco.mjtGeom.mjGEOM_SPHERE, pos=[0.45, 1.15, 0.52], size=[0.24, 0.0, 0.0],
            rgba=(0.20, 0.48, 0.24, 1.0), contype=0, conaffinity=0,
        )
        # Sofa on right side
        spec.worldbody.add_geom(
            name="sofa_base", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[-1.6, -1.95, 0.22], size=[1.05, 0.42, 0.22],
            rgba=(0.32, 0.38, 0.46, 1.0), contype=0, conaffinity=0,
        )
        spec.worldbody.add_geom(
            name="sofa_back", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[-1.6, -2.30, 0.50], size=[1.05, 0.12, 0.32],
            rgba=(0.28, 0.34, 0.42, 1.0), contype=0, conaffinity=0,
        )

        # Dark dining table (Genie Sim table_000, scaled to Q1, matching OpenHLM task_11 dark table)
        table_body = spec.worldbody.add_body(name="home_table", pos=table_xyz * scale)
        for idx, (_, pts, faces) in enumerate(table_meshes):
            mname = f"home_table_{idx}"
            spec.add_mesh(name=mname, uservert=(pts * scale).ravel(), userface=faces.ravel())
            table_body.add_geom(
                name=mname,
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=mname,
                rgba=(0.22, 0.20, 0.19, 1.0),
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

        # White plate on left-center of table
        plate_pos = np.array([-0.04 * scale, 0.15 * scale, top + plate_half_h + 0.001])
        plate_body = spec.worldbody.add_body(name="white_plate", pos=plate_pos)
        plate_body.add_geom(
            name="white_plate_geom",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[plate_radius, plate_half_h, 0.0],
            rgba=(0.94, 0.94, 0.92, 1.0),
            mass=0.2,
            contype=1,
            conaffinity=1,
        )

        # Blue water cup on right-center of table (target object to touch/reach)
        cup_pos = np.array([-0.06 * scale, -0.07 * scale, top + cup_half_h + 0.001])
        cup_body = spec.worldbody.add_body(name="blue_water_cup", pos=cup_pos)
        cup_body.add_geom(
            name="task_target",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[cup_radius, cup_half_h, 0.0],
            rgba=(0.14, 0.46, 0.88, 1.0),
            mass=0.18,
            contype=1,
            conaffinity=1,
        )
        # Water surface & mug handle detail
        cup_body.add_geom(
            name="cup_water_top",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=[0.0, 0.0, cup_half_h * 0.85],
            size=[cup_radius * 0.86, cup_half_h * 0.16, 0.0],
            rgba=(0.68, 0.88, 0.98, 1.0),
            contype=0,
            conaffinity=0,
        )
        cup_body.add_geom(
            name="cup_handle",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0.0, -cup_radius * 1.15, 0.0],
            size=[cup_radius * 0.22, cup_radius * 0.45, cup_half_h * 0.55],
            rgba=(0.14, 0.46, 0.88, 1.0),
            contype=0,
            conaffinity=0,
        )

    return configure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--wam-run", required=True, type=Path)
    parser.add_argument("--wam-checkpoint", required=True, type=Path)
    parser.add_argument("--mimic-run", required=True, type=Path)
    parser.add_argument("--mimic-step", type=int, default=40000)
    parser.add_argument("--p1-processed", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=750)
    parser.add_argument("--spawn-x", type=float, default=-3.2)
    parser.add_argument("--replan-steps", type=int, default=10)
    parser.add_argument("--use-rtc", action="store_true", default=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Explicit seeds; if omitted, draws 2 random seeds")
    parser.add_argument("--num-seeds", type=int, default=2)
    args = parser.parse_args()
    if args.seeds is None:
        rng = random.SystemRandom()
        args.seeds = [rng.randint(1000, 999999) for _ in range(args.num_seeds)]
    print(f"Using random seeds: {args.seeds} | replan_steps={args.replan_steps} ({50.0/args.replan_steps:.1f} Hz) | use_rtc={args.use_rtc}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda:0"
    config = json.loads((args.wam_run / "run.json").read_text())
    saved = torch.load(args.wam_checkpoint, map_location="cpu", weights_only=True)
    dit_block_ids = {int(k.split(".")[2]) for k in saved["expert"] if k.startswith("backbone.transformer_blocks.")}

    wam = CosmosWAM(
        load_backbone(
            Path("/data/kk/models/nvidia/Cosmos-Predict2.5-2B"),
            Path("/data/kk/data/wammimic_downloads_20261116/cosmos_backbone_20261118.json"),
            device,
        ),
        width=config["width"],
        layers=config["layers"],
        unfreeze_blocks=len(dit_block_ids),
        use_proprio=True,
        num_embodiments=config["num_embodiments"],
    ).to(device).eval()
    wam.load_state_dict(saved["expert"], strict=False)
    vae = VideoEncoder("/data/kk/models/wammimic_cosmos25_encoders_20261118")
    gstats = json.loads((args.p1_processed / "canonical_human_5pt_stats.json").read_text())

    # OpenHLM task_11: "Walk forward to the black table, pour the contents from the blue cup into the white plate with the right hand, turn around and walk back."
    with safe_open(str(args.p1_processed / "gripper/prompt_bank_bf16.safetensors"), framework="pt", device="cpu") as f:
        text_cup = f.get_tensor("task_11")[None].to(device)

    mcfg = json.loads((args.mimic_run / "config.json").read_text())
    mdata = Path(mcfg["prepared_manifest"]).parent
    mimic_ckpt = args.mimic_run / "q1" / f"ckpt_{args.mimic_step}.pt"
    policy, meta = load_policy(
        mimic_ckpt, "cpu", interface="human", human_points="five",
        kp_horizon=GOAL_K, goal_anchor=mcfg["goal_anchor"],
    )

    ref_dir = args.out_dir / "reference"
    subset_yaml, rests = prepare_reference(ref_dir, mdata, "q1", np.array([args.spawn_x, 0.0, 0.0]), args.steps + 50)
    cfg = morphmimic_env_cfg(
        "q1", subset_yaml=str(subset_yaml), num_envs=1, play=True, eval=True,
        goal_anchor=mcfg["goal_anchor"], ref_feedback=True,
    )
    cfg.scene.spec_fn = build_home_cup_scene(args.run / "assets", scale=SCALE_Q1, fovy=86.0)

    hz = int(round(50.0 / args.replan_steps))
    new_cases = []
    env = CpuPreviewEnv(cfg, device="cpu")
    try:
        for emb_id, mode_key in ((2, f"home_cup_emb2_rtc{hz}hz"), (1, f"home_cup_emb1_rtc{hz}hz")):
            for seed in args.seeds:
                vname = f"home-table-cup-emb{emb_id}-rtc{hz}hz-seed{seed}.mp4"
                res = run_q1_episode(
                    env, policy, meta, wam, vae, text_cup, gstats, rests,
                    scenario="home-table-cup-q1", mode=mode_key, seed=seed,
                    video_path=args.out_dir / vname, steps=args.steps,
                    emb_id=emb_id, replan_steps=args.replan_steps, use_rtc=args.use_rtc,
                )
                new_cases.append(res)
                summary = {k: v for k, v in res.items() if k not in ("planning_calls", "physics")}
                print(json.dumps(summary), flush=True)
    finally:
        env.close()

    # Prepend new home-cup cases into manifest.json while keeping all existing cases
    manifest_path = args.out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    existing = [c for c in manifest.get("cases", []) if c.get("scenario") != "home-table-cup-q1"]
    manifest["cases"] = new_cases + existing
    write_json(manifest_path, manifest)
    print(f"Updated manifest with {len(new_cases)} home-cup cases. Total={len(manifest['cases'])}", flush=True)


if __name__ == "__main__":
    main()
