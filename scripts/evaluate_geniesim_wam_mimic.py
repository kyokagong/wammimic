"""Evaluate WAM plus Q1 Mimic on published Genie Sim tabletop layouts."""

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
from pxr import Gf, Usd, UsdGeom
from scipy.spatial.transform import Rotation, Slerp
from scipy.optimize import least_squares
from safetensors.torch import load_file
import torch

from morphmimic.constants import GOAL_K, HUMAN_FIVE_IDX, NUM_HUMAN, PROPRIO_DIM
from morphmimic.env.env_cfg import COMMAND_NAME, morphmimic_env_cfg
from morphmimic.eval.policy_loader import load_policy
from morphmimic.model.human import pack_human_goal
from morphmimic.model.ref_feedback import REF_FRAMES, REF_SLOTS, global_ref_feedback
from morphmimic.model.state import anchor_quat, join_state, split_state
from morphmimic.train.obs_history import ObsHistory
from preview.cpu_backend import CpuPreviewEnv
from wammimic.data_pipeline.cache import VideoEncoder
from wammimic.data_pipeline.common import sha256, write_json
from wammimic.data_pipeline.geometry import LEG_LENGTH, matrices
from wammimic.data_pipeline.human_ego import human_features, human_history
from wammimic.training.evaluation import rotation6d
from wammimic.training.model import CosmosWAM, load_backbone


TASKS = (
    ("press-button", "press_button_g2", "1038473124_41f905e0", "benchmark_button_1dbfb257"),
    ("matching-block-contact", "pick_block_matching_reference_color_g2", "1472563149_665eedea",
     "benchmark_building_blocks_e44d4bc0"),
)
SLOTS = (0, *REF_SLOTS)
DT = .02
SPAWN = np.array([-.43, 0., 0.])
COLORS = {"white": (.86, .84, .8, 1), "blue": (.14, .43, .91, 1), "red": (.92, .19, .22, 1),
          "purple": (.59, .21, .75, 1), "brown": (.55, .34, .18, 1)}
SCALE = LEG_LENGTH["q1"] / LEG_LENGTH["g1"]


def usd_geometry(file):
    stage = Usd.Stage.Open(str(file))
    if not stage or UsdGeom.GetStageUpAxis(stage) != "Z" or UsdGeom.GetStageMetersPerUnit(stage) != 1:
        raise ValueError(f"Expected metre Z-up asset {file}")
    cache = UsdGeom.XformCache()
    meshes = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        transform = cache.GetLocalToWorldTransform(prim)
        points = np.array([transform.Transform(Gf.Vec3d(*point)) for point in mesh.GetPointsAttr().Get()])
        indices = list(mesh.GetFaceVertexIndicesAttr().Get())
        faces = []
        start = 0
        for count in mesh.GetFaceVertexCountsAttr().Get():
            faces.extend((indices[start], indices[start + i], indices[start + i + 1]) for i in range(1, count - 1))
            start += count
        meshes.append(("part_00" in str(prim.GetPath()), points, np.asarray(faces)))
    if not meshes:
        raise ValueError(f"No composed USD geometry in {file}")
    return meshes


def scene_callback(scene, assets, target):
    files = {p.parent.name: p for p in assets.rglob("Aligned.usda")}
    objects = [(key, row, usd_geometry(files[row["usd"]])) for key, row in scene["layout"].items()]
    _, table, surfaces = next(item for item in objects if item[1]["usd"] == "table_000")
    source_top = table["xyz"][2] + max(points[:, 2].max() for _, points, _ in surfaces)
    top = source_top * SCALE

    def configure(spec):
        pitch = np.deg2rad(27)
        torso = spec.body("robot/torso_link")
        torso.add_camera(name="wam_head", pos=[.085, 0., .23], fovy=72,
                         xyaxes=[0., -1., 0., np.sin(pitch), 0., np.cos(pitch)])
        for key, row, meshes in objects:
            table_asset = row["usd"] == "table_000"
            button = row["usd"].startswith("benchmark_button")
            is_target = key == target
            position = np.array(row["xyz"], dtype=float) * SCALE
            factor = SCALE if table_asset else 1.
            if not table_asset:
                # shortcut: fix objects at their settled height, upgrade for grasp or drop dynamics.
                bottom = min(points[:, 2].min() for _, points, _ in meshes)
                clearance = 0. if button else max(0., row["xyz"][2] + bottom - source_top)
                position[2] = top - bottom + clearance
            quat = np.asarray(row["xyzw"])[[3, 0, 1, 2]]
            body = spec.worldbody.add_body(name=key, pos=position, quat=quat)
            for index, (cap, points, faces) in enumerate(meshes):
                name = f"{key}_{index}"
                spec.add_mesh(name=name, uservert=(points * factor).ravel(), userface=faces.ravel())
                parent = body
                if cap:
                    parent = body.add_body(name="button_cap", gravcomp=1.)
                    parent.add_joint(name="button_travel", type=mujoco.mjtJoint.mjJNT_SLIDE,
                                     axis=[0, 0, -1], limited=True, range=[0., .005], stiffness=180., damping=1.)
                    # The source USD disables contact between the two slider bodies.
                    spec.add_exclude(name="button_internal", bodyname1=key, bodyname2="button_cap")
                color = COLORS.get(row["description"].get("color"), COLORS["white"])
                if button and not cap:
                    color = COLORS["white"]
                parent.add_geom(name="task_target" if is_target and (cap or not button) else name,
                                type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name, rgba=color,
                                mass=.2 if cap else .05, contype=int(not table_asset), conaffinity=int(not table_asset))
        spec.worldbody.add_geom(name="genie_table_collision", type=mujoco.mjtGeom.mjGEOM_BOX,
                                pos=[0., 0., top - .025], size=[.30 * SCALE, .50 * SCALE, .025],
                                rgba=[0., 0., 0., 0.], contype=1, conaffinity=1)
        for key in spec.keys:
            if len(key.qpos) and any(row["usd"].startswith("benchmark_button") for _, row, _ in objects):
                key.qpos = np.r_[key.qpos, 0.]
                if len(key.qvel):
                    key.qvel = np.r_[key.qvel, 0.]

    return configure


def reference(run, data, frames):
    folder = run / "reference"
    folder.mkdir(exist_ok=True)
    with np.load(data / "q1/sprint1_subject2.npz") as original:
        arrays = {k: np.repeat(original[k][:1], frames, axis=0) if original[k].ndim > 1 else original[k]
                  for k in original.files}
        arrays["body_pos_w"] += SPAWN
        for name in ("joint_vel", "body_lin_vel_w", "body_ang_vel_w"):
            arrays[name].fill(0.)
        np.savez(folder / "neutral.npz", **arrays)
    with np.load(data / "human50/q1/sprint1_subject2.npz") as human:
        human_pos = human["human_pos_w"][0, list(HUMAN_FIVE_IDX)].copy()
        human_rot = matrices(human["human_quat_w"][0, list(HUMAN_FIVE_IDX)])
    write_json(folder / "subset.yaml", {"root": str(folder), "clips": ["neutral.npz"]})
    return arrays["body_pos_w"][0, list(SLOTS)], matrices(arrays["body_quat_w"][0, list(SLOTS)]), human_pos, human_rot


def optical_stance(protocol):
    """Find a stationary upright stance from training mocap, without using the simulated task."""
    best = None
    for entry in protocol["episodes"]:
        if entry["split"] != "train" or entry["source_id"] != 0:
            continue
        path = Path(protocol["sources"][0]["root"]) / (
            f'wam_lerobot_v2/human_ego/labels/episode_{entry["episode_index"]:06d}.npz')
        with np.load(path) as poses:
            p = poses["poses_world"].astype(float)
            quality = poses["quality_valid"]
            leg = float(poses["human_leg_length"])
            xyz = p[..., :3]
            offset = xyz - xyz[:, :1]
            feet = offset[:, 1:3]
            wrists = offset[:, 3:]
            speed = np.linalg.norm(np.diff(xyz, axis=0), axis=-1).mean(1)
            cost = (speed + 2 * np.abs(feet[:-1, :, 2] + leg).mean(1)
                    + np.abs(feet[:-1, 0, 2] - feet[:-1, 1, 2])
                    + 2 * np.abs(wrists[:-1, :, 2] + .08).mean(1))
            valid = quality[:-1] & quality[1:] & np.isfinite(p[:-1]).all((1, 2))
            if not valid.any():
                continue
            i = int(np.argmin(np.where(valid, cost, np.inf)))
            if best is None or float(cost[i]) < best[0]:
                best = float(cost[i]), p[i], leg, entry["episode_index"], i
    if best is None:
        raise ValueError("No measured training stance")
    _, pose, leg, episode, frame = best
    left = (pose[1, :3] - pose[2, :3]) + (pose[3, :3] - pose[4, :3])
    left[2] = 0.
    left /= np.linalg.norm(left)
    up = np.array([0., 0., 1.])
    canonical = np.stack((np.cross(left, up), left, up), axis=1)
    observed = matrices(pose[:, 3:])
    basis = canonical.T @ observed[0]
    return basis, (pose[:, :3] - pose[0, :3]) @ observed[0], observed[0].T @ observed, leg, [episode, frame]


def block_targets(block, actual, rests, optical, anchor_xy=None, max_lead_m=.08):
    basis, source_pos, source_rot, source_leg, _ = optical
    robot_pos, robot_rot, human_pos, human_rot = rests
    current_pos, current_rot = actual
    yaw = np.arctan2(current_rot[0, 1, 0], current_rot[0, 0, 0])
    heading = Rotation.from_euler("z", yaw).as_matrix()
    rest_frame = heading @ basis
    # human_features() projects into the full 3D pelvis frame (including pitch/roll),
    # so unproject with the robot's full 3D pelvis delta while keeping neutral offsets yaw-only.
    pelvis_delta = current_rot[0] @ robot_rot[0].T
    frame = pelvis_delta @ basis
    position = block[:, :15].reshape(-1, 5, 3).copy()
    position[:, :3] *= source_leg
    rot, valid = rotation6d(block[:, 15:].reshape(-1, 5, 6))
    if not np.isfinite(position).all() or not valid.all():
        raise ValueError("Nonfinite or degenerate WAM target")
    delta_human = position @ frame.T - source_pos @ rest_frame.T
    # Preserve bounded forward tracking lead on XY while anchoring Z to standing height
    # so tracking lag or slight crouching is never baked into a runaway drift loop.
    origin_xy = current_pos[0, :2].copy()
    if anchor_xy is not None:
        lead = anchor_xy - origin_xy
        norm = float(np.linalg.norm(lead))
        if norm > max_lead_m:
            lead = lead * (max_lead_m / norm)
        origin_xy = origin_xy + lead
    robot_origin = np.array([origin_xy[0], origin_xy[1], robot_pos[0, 2]])
    human_origin = np.array([origin_xy[0], origin_xy[1], human_pos[0, 2]])
    scale_leg = LEG_LENGTH["q1"] / source_leg
    scale_arm = float(np.linalg.norm(robot_pos[3] - robot_pos[4]) / np.linalg.norm(source_pos[3] - source_pos[4]))
    delta_robot = np.empty_like(delta_human)
    delta_robot[:, :3] = delta_human[:, :3] * scale_leg
    delta_robot[:, 3:] = (delta_human[:, :1] * scale_leg
                          + (delta_human[:, 3:] - delta_human[:, :1]) * scale_arm)
    change = (frame @ rot) @ (rest_frame @ source_rot).swapaxes(-1, -2)
    human = (human_pos - human_pos[0]) @ heading.T + delta_human + human_origin
    goal = (robot_pos - robot_pos[0]) @ heading.T + delta_robot + robot_origin
    return (human, change @ (heading @ human_rot)), (goal, change @ (heading @ robot_rot))


def interpolate(first, block):
    pos, rot = block
    keys = np.arange(len(pos) + 1) * .1
    stamps = np.arange(51) * DT
    starts, orientations = first
    values = np.concatenate((starts[None], pos), axis=0)
    full_p = np.stack([np.interp(stamps, keys, values[:, i, j]) for i in range(5) for j in range(3)], axis=-1)
    full_r = np.stack([Slerp(keys, Rotation.from_matrix(np.concatenate((orientations[i:i+1], rot[:, i]), axis=0)))(stamps).as_matrix()
                       for i in range(5)], axis=1)
    return full_p.reshape(-1, 5, 3), full_r


def tensor(value, device):
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def wxyz(rotation):
    return Rotation.from_matrix(rotation.reshape(-1, 3, 3)).as_quat()[:, [3, 0, 1, 2]].reshape(*rotation.shape[:-2], 4)


def button_reference(env, rests, steps, walk_clip, advance=.18, press_depth=.034):
    """Adapt a recorded approach to the table and append a joint-limit checked hand press."""
    model = env.sim.mj_model
    data = mujoco.MjData(model)
    data.qpos[:] = env.reference_qpos(0)
    mujoco.mj_forward(model, data)
    wrist = model.body("robot/right_wrist_yaw_link").id
    hand = model.geom("robot/right_wrist_yaw_link_0_collision").id
    button = model.geom("task_target").id
    mesh = int(model.geom_dataid[button])
    start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
    vertices = model.mesh_vert[start:start+count] @ data.geom_xmat[button].reshape(3, 3).T
    button_xy = data.geom_xpos[button, :2].copy()
    top = float(data.geom_xpos[button, 2] + vertices[:, 2].max())
    arm = [model.joint("robot/right_" + name + "_joint").id for name in
           ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")]
    address = model.jnt_qposadr[arm]
    neutral_arm = data.qpos[address].copy()
    low, high = env.native_joint_limits.numpy()[[list(env.joint_qadr).index(i) for i in address]].T
    data.qpos[0] += advance
    arm_seed = np.clip(neutral_arm, low + 1e-6, high - 1e-6)
    positions, rotations = [], []
    for height in (top + .09, top + .012, top - press_depth):
        target = np.r_[button_xy, height]

        def error(q):
            data.qpos[address] = q
            mujoco.mj_forward(model, data)
            geom_rotation = data.geom_xmat[hand].reshape(3, 3)
            tip = data.geom_xpos[hand].copy()
            tip[2] -= np.abs(geom_rotation[2]) @ model.geom_size[hand]
            return np.r_[tip - target, .006 * (q - arm_seed)]

        trial = least_squares(error, arm_seed, bounds=(low, high), max_nfev=120)
        gap = np.linalg.norm(error(trial.x)[:3])
        if gap > .025:
            raise ValueError(f"Button reference outside arm reach: residual {gap:.3f} m")
        arm_seed = trial.x
        positions.append(data.xpos[wrist].copy())
        rotations.append(data.xmat[wrist].reshape(3, 3).copy())

    def smooth(t, start, stop):
        u = np.clip((t - start) / (stop - start), 0., 1.)
        return u**3 * (10. + u * (-15. + 6. * u))

    xyz, orientation, human_xyz, human_orientation = rests
    t = np.arange(steps + GOAL_K + 1) * DT
    robot = np.repeat(xyz[None], len(t), axis=0)
    robot_rot = np.repeat(orientation[None], len(t), axis=0)
    with np.load(walk_clip) as walk:
        source = walk["body_pos_w"][:, list(SLOTS[:3])].astype(float)
        source_rot = matrices(walk["body_quat_w"][:, list(SLOTS[:3])])
    start, finish = 238, 365
    # Preserve the ordering of swings and the vertical toe clearance while shortening the steps.
    timing = np.arange(len(t)) * ((finish - start) / 111)
    frame = np.clip(np.rint(start + timing).astype(int), start, finish)
    shift = source[frame] - source[0]
    shift[..., 0] *= advance / (source[finish, 0, 0] - source[0, 0, 0])
    shift[..., 1] *= .45
    robot[:, :3] += shift
    robot_rot[:, :3] = source_rot[frame]
    robot[:, 3:] += shift[:, :1]

    approach = smooth(t, .28, 1.35)
    second = smooth(t, 1.80, 2.35)
    third = smooth(t, 2.35, 3.10)
    home = xyz[4] + shift[-1, 0]
    robot[:, 4] += (approach[:, None] * (positions[0] - home)
                    + second[:, None] * (positions[1] - positions[0])
                    + third[:, None] * (positions[2] - positions[1]))
    quaternion_path = Rotation.from_matrix(np.stack([orientation[4], *rotations]))
    robot_rot[:, 4] = Slerp(np.arange(4), quaternion_path)(approach + second + third).as_matrix()
    human = (human_xyz + SPAWN)[None] + (robot - xyz[None]) * 1.65
    human_rot = robot_rot @ orientation.swapaxes(-1, -2) @ human_orientation
    return (human, human_rot), (robot, robot_rot)


def actor_targets(state, human, robot, index):
    device = state.device
    human_pos = np.zeros((1, GOAL_K, NUM_HUMAN, 3))
    human_q = np.zeros((1, GOAL_K, NUM_HUMAN, 4))
    human_q[..., 0] = 1.
    human_pos[0][:, HUMAN_FIVE_IDX, :] = human[0][index + 1:index + 1 + GOAL_K]
    human_q[0][:, HUMAN_FIVE_IDX, :] = wxyz(human[1][index + 1:index + 1 + GOAL_K])
    goal = pack_human_goal(tensor(human_pos, device), tensor(human_q, device), anchor_quat(state))
    ref = split_state(state[:, None].repeat(1, REF_FRAMES, 1))
    ref.body_pos[0][:, SLOTS, :] = tensor(robot[0][index:index + REF_FRAMES], device)
    ref.body_quat[0][:, SLOTS, :] = tensor(wxyz(robot[1][index:index + REF_FRAMES]), device)
    feedback = global_ref_feedback(state, join_state(ref), torch.tensor([REF_FRAMES], device=device))
    return {"goal_kp": goal, "ref_feedback": feedback}


def q1_direct_targets(block, actual, rests, anchor_xy=None, max_lead_m=.08):
    """Exact inverse of sim.py human_features() in Q1's current t0 full 3D pelvis frame with bounded XY lead and standing Z anchor."""
    robot_rest_pos, robot_rest_rot, human_rest_pos, human_rest_rot = rests
    current_pos, current_rot = actual
    r0 = current_rot[0]
    pos_local = block[:, :15].reshape(-1, 5, 3).copy()
    pos_local[:, :3] *= LEG_LENGTH["q1"]
    rot_local, valid = rotation6d(block[:, 15:45].reshape(-1, 5, 6))
    if not np.isfinite(pos_local).all() or not valid.all():
        raise ValueError("Nonfinite or degenerate P1 WAM target")
    origin_xy = current_pos[0, :2].copy()
    if anchor_xy is not None:
        lead = anchor_xy - origin_xy
        norm = float(np.linalg.norm(lead))
        if norm > max_lead_m:
            lead = lead * (max_lead_m / norm)
        origin_xy = origin_xy + lead
    origin = np.array([origin_xy[0], origin_xy[1], robot_rest_pos[0, 2]])
    goal = pos_local @ r0.T + origin
    goal_rot = r0[None, None] @ rot_local
    human = (human_rest_pos + SPAWN)[None] + (goal - robot_rest_pos[None]) * 1.65
    human_rot = goal_rot @ robot_rest_rot.swapaxes(-1, -2) @ human_rest_rot
    return (human, human_rot), (goal, goal_rot)


@torch.inference_mode()
def predict(wam, vae, frames, text, generator, mean, scale, *, proprio=None, ee_hist=None, embodiment_id=None,
            prefix_target=None, prefix_weights=None):
    # The cached training contract has four measured history images; repeat the final one for unused latents.
    history = [frames[0], *frames, *([frames[-1]] * 12)]
    pixels = torch.from_numpy(np.stack(history)).permute(3, 0, 1, 2)[None].float() / 127.5 - 1.
    video = vae.encode(pixels)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if getattr(wam, "use_proprio", False):
            emb_t = torch.tensor([int(embodiment_id)], dtype=torch.int64, device="cuda:0")
            p_t = torch.as_tensor(proprio[None], dtype=torch.float32, device="cuda:0")
            e_t = torch.as_tensor(ee_hist[None], dtype=torch.float32, device="cuda:0")
            context = wam.encode_history(video, text, proprio=p_t, ee_hist=e_t, embodiment_id=emb_t)
            noise = torch.randn((1, 10, 47), device="cuda:0", generator=generator)
            pt_t = None if prefix_target is None else torch.as_tensor(prefix_target[None], dtype=torch.float32, device="cuda:0")
            pw_t = None if prefix_weights is None else torch.as_tensor(prefix_weights, dtype=torch.float32, device="cuda:0")
            return wam.sample(context, noise, 16, embodiment_id=emb_t, prefix_target=pt_t, prefix_weights=pw_t)[0].cpu().numpy() * scale + mean
        context = wam.encode_history(video, text)
        noise = torch.randn((1, 10, 45), device="cuda:0", generator=generator)
        return wam.sample(context, noise, 16)[0].cpu().numpy() * scale + mean


def render_panel(renderer, mjdata, camera, image, target_pos, status):
    renderer.update_scene(mjdata, camera=camera)
    if target_pos is not None:
        for i, point in enumerate(target_pos):
            scene = renderer.scene
            if scene.ngeom == scene.maxgeom:
                break
            color = ([1., .48, .12, 1.] if i < 3 else [.1, .85, 1., 1.])
            mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                               np.full(3, .014), point, np.eye(3).ravel(), color)
            scene.ngeom += 1
    panel = Image.fromarray(renderer.render())
    panel.paste(Image.fromarray(image).resize((200, 200)), (12, 12))
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 482, 960, 540), fill=(19, 24, 36))
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    draw.text((15, 489), status, font=font, fill=(245, 248, 255))
    draw.text((15, 514), "Inset: head camera RGB    Markers: pelvis/ankles and wrist targets", font=font, fill=(205, 213, 227))
    return np.asarray(panel)


@torch.inference_mode()
def rollout(env, policy, meta, wam, vae, text, stats, rests, optical, scenario, seed, path, steps,
            mode, scripted=None, state_stats=None, emb_id=0):
    state_ids = torch.arange(1, device=env.device)
    command = env.command_manager.get_term(COMMAND_NAME)
    command.assign_clips(state_ids, torch.zeros_like(state_ids))
    obs, _ = env.reset(env_ids=state_ids)
    history = ObsHistory(1, meta.hist_len, PROPRIO_DIM, env.device)
    history.reset(state_ids, obs["policy"])
    generator = torch.Generator(device="cuda:0").manual_seed(seed) if mode == "wam_mimic" else None
    model = env.sim.mj_model
    data = env.native_data
    model.vis.global_.offwidth, model.vis.global_.offheight = 960, 540
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [-.25, 0., .32]
    camera.distance, camera.azimuth, camera.elevation = 1.58, 128., -18.
    target = model.geom("task_target").id
    hand_geoms = {g for g in range(model.ngeom) if "robot/" in model.body(model.geom_bodyid[g]).name
                  and any(term in model.body(model.geom_bodyid[g]).name for term in ("wrist", "hand"))}
    if not hand_geoms:
        raise ValueError("Q1 scene has no hand contact geometry")
    button_qadr = None if scenario != "press-button" else int(model.jnt_qposadr[model.joint("button_travel").id])
    frames = deque(maxlen=4)
    updates, contact_steps = [], 0
    closest, depth, reason = np.inf, 0., "timeout"
    human = robot = None
    if mode == "reference":
        human, robot = scripted
    human_start = rests[2] + SPAWN
    trajectories = []
    if mode == "wam_mimic":
        mean, scale = np.asarray(stats["mean"]), np.maximum(np.asarray(stats["std"]), 1e-6)
        if state_stats is not None:
            s_mean, s_scale = np.asarray(state_stats["mean"]), np.maximum(np.asarray(state_stats["std"]), 1e-6)
    poses_50hz = []
    touches_50hz = []
    hand_sides = [{i for i in range(model.ngeom) if f"/{side}_wrist" in model.geom(i).name} for side in ("left", "right")]
    if mode == "reference_retargeted":
        reference_robot_pos, reference_robot_rot = scripted[1]
        basis, source_pos, source_rot, source_leg, _ = optical
        scale_leg = LEG_LENGTH["q1"] / source_leg
        scale_arm = float(np.linalg.norm(rests[0][3] - rests[0][4]) / np.linalg.norm(source_pos[3] - source_pos[4]))
        delta_robot = reference_robot_pos - rests[0][None]
        delta_human = np.empty_like(delta_robot)
        delta_human[:, :3] = delta_robot[:, :3] / scale_leg
        delta_human[:, 3:] = (delta_robot[:, :1] / scale_leg
                              + (delta_robot[:, 3:] - delta_robot[:, :1]) / scale_arm)
        opt_rest_world = source_pos @ basis.T + np.array([rests[0][0, 0], rests[0][0, 1], source_leg])
        reference_pos = opt_rest_world[None] + delta_human
        optical_rot = reference_robot_rot @ rests[1].swapaxes(-1, -2) @ (basis @ source_rot)
        reference_poses = np.concatenate((reference_pos, wxyz(optical_rot)), axis=-1)
    label_by_mode = {
        "hold": "hold baseline", "wam_mimic": "WAM + Mimic", "reference": "reference five points + Mimic",
        "reference_sampled": "sampled reference + Mimic", "reference_retargeted": "reference via WAM retarget + Mimic",
    }
    with ExitStack() as stack:
        head = stack.enter_context(mujoco.Renderer(model, height=224, width=224))
        wide = stack.enter_context(mujoco.Renderer(model, height=540, width=960))
        video = stack.enter_context(subprocess.Popen(
            ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", "960x540", "-r", "25", "-i", "pipe:0", "-c:v", "libx264", "-threads", "2", "-pix_fmt", "yuv420p", str(path)],
            stdin=subprocess.PIPE))
        last_image = None
        for k in range(steps):
            state = env.state()
            parts = split_state(state)
            actual_pos = parts.body_pos[0, list(SLOTS)].cpu().numpy()
            actual_quat = parts.body_quat[0, list(SLOTS)].cpu().numpy()
            actual_rot = matrices(actual_quat)
            poses_50hz.append(np.concatenate((actual_pos, actual_quat), axis=-1))
            t_force = np.zeros(2, dtype=np.float32)
            wrench = np.zeros(6)
            for ci, c in enumerate(data.contact[:data.ncon]):
                ca, cb = int(c.geom[0]), int(c.geom[1])
                for s_i, gset in enumerate(hand_sides):
                    if (ca == target and cb in gset) or (cb == target and ca in gset):
                        mujoco.mj_contactForce(model, data, ci, wrench)
                        t_force[s_i] += max(0., float(wrench[0]))
            touches_50hz.append((t_force >= .5).astype(np.float32))
            if k % 5 == 0:
                head.update_scene(data, camera="wam_head")
                last_image = head.render().copy()
                frames.append(last_image)
            if mode in ("wam_mimic", "reference_sampled", "reference_retargeted") and len(frames) == 4 and (k - 15) % 10 == 0:
                started = perf_counter()
                future = slice(k + 5, k + 51, 5)
                old = k - updates[-1]["control_step"] if updates else 0
                if mode == "reference_sampled":
                    human_block, robot_block = ((pos[future], rot[future]) for pos, rot in scripted)
                else:
                    anchor_xy = None if robot is None else robot[0][old, 0, :2]
                    if mode == "reference_retargeted":
                        # An ideal closed-loop WAM predicts future targets in the robot's current 3D pelvis frame.
                        anchor_pelvis_delta = actual_rot[0] @ rests[1][0].T
                        anchor_opt_rot = anchor_pelvis_delta @ basis
                        origin_xy = actual_pos[0, :2].copy()
                        if anchor_xy is not None:
                            lead = anchor_xy - origin_xy
                            norm = float(np.linalg.norm(lead))
                            if norm > .08:
                                lead = lead * (.08 / norm)
                            origin_xy = origin_xy + lead
                        scale_leg = LEG_LENGTH["q1"] / source_leg
                        anchor_pelvis_opt = np.array([
                            rests[0][0, 0] + (origin_xy[0] - rests[0][0, 0]) / scale_leg,
                            rests[0][0, 1] + (origin_xy[1] - rests[0][0, 1]) / scale_leg,
                            source_leg,
                        ])
                        anchor_pose = reference_poses[k].copy()
                        anchor_pose[0, :3] = anchor_pelvis_opt
                        anchor_pose[0, 3:] = wxyz(anchor_opt_rot)
                        prediction = human_features(reference_poses[future], anchor_pose, source_leg)
                        assert prediction.shape == (10, 45)
                    elif getattr(wam, "use_proprio", False):
                        p_arr = np.asarray(poses_50hz, dtype=np.float64)
                        t_arr = np.arange(len(p_arr), dtype=np.float64) * DT
                        hist_idx = np.maximum(np.array([[k - 15, k - 10, k - 5, k]], dtype=int), 1)
                        h75 = human_history(p_arr, t_arr, hist_idx, LEG_LENGTH["q1"])[0]
                        st75 = (h75 - s_mean[:75]) / s_scale[:75]
                        ee_h4 = np.zeros((4, 4), dtype=np.float32)
                        if len(s_mean) == 77:
                            t_hist = np.asarray(touches_50hz, dtype=np.float32)[hist_idx[0]]
                            ee_h4[:, :2] = (t_hist - s_mean[75:77]) / s_scale[75:77]
                        prediction = predict(wam, vae, list(frames), text, generator, mean, scale,
                                             proprio=st75.astype(np.float32), ee_hist=ee_h4, embodiment_id=emb_id)
                        human_block, robot_block = q1_direct_targets(prediction, (actual_pos, actual_rot), rests, anchor_xy=anchor_xy)
                    else:
                        prediction = predict(wam, vae, list(frames), text, generator, mean, scale)
                        human_block, robot_block = block_targets(prediction, (actual_pos, actual_rot), rests, optical,
                                                                 anchor_xy=anchor_xy)
                    if mode == "reference_retargeted":
                        human_block, robot_block = block_targets(prediction, (actual_pos, actual_rot), rests, optical,
                                                                 anchor_xy=anchor_xy)
                start_h = (human_start, rests[3]) if human is None else (human[0][old], human[1][old])
                start_r = (actual_pos, actual_rot) if robot is None else (robot[0][old], robot[1][old])
                human, robot = interpolate(start_h, human_block), interpolate(start_r, robot_block)
                updates.append({"control_step": k, "inference_s": perf_counter() - started})
            if robot is None:
                targets = actor_targets(state, (np.repeat(human_start[None], 11, 0), np.repeat(rests[3][None], 11, 0)),
                                        (np.repeat(rests[0][None], 11, 0), np.repeat(rests[1][None], 11, 0)), 0)
                markers = None
            else:
                index = k if mode == "reference" else k - updates[-1]["control_step"]
                targets = actor_targets(state, human, robot, index)
                markers = robot[0][index]
            pairs = [(int(c.geom[0]), int(c.geom[1])) for c in data.contact[:data.ncon] if c.dist <= .001]
            contact = any((a == target and b in hand_geoms) or (b == target and a in hand_geoms) for a, b in pairs)
            contact_steps += int(contact)
            distance = float(np.linalg.norm(actual_pos[3:] - data.geom_xpos[target], axis=-1).min())
            closest = min(closest, distance)
            if button_qadr is not None and contact:
                depth = max(depth, float(data.qpos[button_qadr]))
            action = policy(history.flat(), targets)
            if not bool(torch.isfinite(action).all()):
                raise ValueError("Mimic produced nonfinite action")
            trajectories.append({"step": k, "wrist_target_distance_m": distance, "hand_contact": contact,
                                 "button_travel_m": None if button_qadr is None else float(data.qpos[button_qadr]),
                                 "nonfloor_contacts": [[model.geom(a).name, model.geom(b).name] for a, b in pairs
                                                       if not any(model.geom(i).name in ("terrain", "floor") for i in (a, b))],
                                 "task_object_position_m": data.geom_xpos[target].tolist(),
                                 "target_positions_m": None if markers is None else markers.tolist(),
                                 "actual_positions_m": actual_pos.tolist(), "canonical_action": action[0].tolist()})
            reached = depth >= .003 if button_qadr is not None else contact_steps > 0
            if k % 2 == 0 or reached:
                label = f"{scenario} / {label_by_mode[mode]} / seed {seed}   {k * DT:.2f}s   closest {closest*100:.1f}cm   contact {contact_steps}"
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
    path.with_suffix(".jsonl").write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in trajectories))
    return {"video": path.name, "scenario": scenario, "seed": seed, "mode": mode,
            "success": reason == "success", "termination": reason, "steps": k + 1, "min_wrist_distance_m": closest,
            "hand_contact_steps": contact_steps, "button_max_travel_m": None if button_qadr is None else depth,
            "planning_calls": updates, "physics": diagnostics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--wam-run", required=True, type=Path)
    parser.add_argument("--wam-checkpoint", required=True, type=Path)
    parser.add_argument("--mimic-run", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--out-name", type=str, default=None)
    parser.add_argument("--p1-sim-stats", type=Path, default=None)
    parser.add_argument("--embodiment-id", type=int, default=None)
    button = parser.add_mutually_exclusive_group()
    button.add_argument("--button-reference", action="store_true")
    button.add_argument("--button-comparison", action="store_true")
    button.add_argument("--button-retarget-ablation", action="store_true")
    args = parser.parse_args()
    if args.steps < 20 or len(set(args.seeds)) != len(args.seeds):
        parser.error("At least 20 steps and unique seeds are required")
    run = args.run
    source = run / "genie_sim/source/geniesim_scene_agent/examples/tasks"
    out = run / (args.out_name or ("button-retarget-ablation" if args.button_retarget_ablation
                 else "button-comparison" if args.button_comparison else "replays"))
    out.mkdir(exist_ok=True)
    config = json.loads((args.wam_run / "run.json").read_text())
    is_p1 = config.get("schema") == "wam-p1-multimodal-v1"
    frozen_mimic = json.loads((args.mimic_run / "config.json").read_text())
    if not frozen_mimic["ref_feedback"]:
        raise ValueError("Missing Global + Ref. interface")
    if is_p1:
        sim_stats_path = args.p1_sim_stats or (Path(config["p1_processed_root"]) / "sim/wam_lerobot_v2/q1_stub/meta/stats.json")
        sim_stats = json.loads(sim_stats_path.read_text())
        protocol = None
    else:
        protocol = json.loads(Path(config["protocol"]).read_text())
        if sha256(config["protocol"]) != config["protocol_sha256"]:
            raise ValueError("Unexpected checkpoint data")
        sim_stats = None
    fp = args.mimic_run / "q1/ckpt_12000.pt"
    device = "cuda:0"
    torch.manual_seed(0)
    policy, meta = load_policy(fp, "cpu", interface="human", human_points="five", kp_horizon=GOAL_K,
                               goal_anchor=frozen_mimic["goal_anchor"])
    assert meta.robot == "q1" and meta.iteration == 12000
    mimic_data = Path(frozen_mimic["prepared_manifest"]).parent
    rests = reference(run, mimic_data, args.steps + 20)
    if args.button_reference or args.button_retarget_ablation:
        scenario, task, layout, target = TASKS[0]
        scene_path = source / task / layout / "scene_info.json"
        scene = json.loads(scene_path.read_text())
        cfg = morphmimic_env_cfg("q1", subset_yaml=str(run / "reference/subset.yaml"), num_envs=1, play=True, eval=True,
                                 goal_anchor=frozen_mimic["goal_anchor"], ref_feedback=True)
        cfg.scene.spec_fn = scene_callback(scene, run / "assets", target)
        env = CpuPreviewEnv(cfg, device="cpu")
        try:
            goals = button_reference(env, rests, args.steps + 50, mimic_data / "q1/walk1_subject5.npz")
            optical = optical_stance(protocol) if args.button_retarget_ablation else None
            modes = ("reference", "reference_sampled", "reference_retargeted") if optical is not None else ("reference",)
            cases = []
            for mode in modes:
                video = out / f"press-button-{mode.replace('_', '-')}.mp4"
                result = rollout(env, policy, meta, None, None, None, None, rests, optical, scenario, 0,
                                 video, args.steps, mode, scripted=goals)
                cases.append(result)
                print(json.dumps(result), flush=True)
        finally:
            env.close()
        if args.button_retarget_ablation:
            write_json(out / "manifest.json", {"scenario": scenario, "scene_sha256": sha256(scene_path),
                       "mimic_checkpoint_sha256": sha256(fp), "optical_calibration": optical[-1], "cases": cases})
        else:
            result["mimic_checkpoint_sha256"] = sha256(fp)
            write_json(out / "press-button-reference.json", result)
        return
    saved = torch.load(args.wam_checkpoint, map_location="cpu", weights_only=True)
    if is_p1:
        dit_block_ids = {int(k.split(".")[2]) for k in saved["expert"] if k.startswith("backbone.transformer_blocks.")}
        unfreeze_n = len(dit_block_ids)
        wam = CosmosWAM(
            load_backbone(config["backbone"], config["backbone_manifest"], device),
            width=config["width"], layers=config["layers"],
            unfreeze_blocks=unfreeze_n, use_proprio=True, num_embodiments=config["num_embodiments"],
        ).to(device).eval()
    else:
        wam = CosmosWAM(load_backbone(config["backbone"], config["backbone_manifest"], device), train_backbone=False).to(device).eval()
    assert saved["step"] == 5000 and saved["expert"].keys() == {k for k, p in wam.named_parameters() if p.requires_grad}
    wam.load_state_dict(saved["expert"], strict=False)
    vae = VideoEncoder("/data/kk/models/wammimic_cosmos25_encoders_20261118")
    bank = load_file(run / "text/prompt_bank_bf16.safetensors", device=device)
    optical = None if is_p1 else optical_stance(protocol)
    act_stats = sim_stats["action"] if is_p1 else protocol["statistics"]["action"]
    st_key = "observation.state_5pt_hist" if (is_p1 and "observation.state_5pt_hist" in sim_stats) else "observation.state"
    st_stats = sim_stats[st_key] if is_p1 else None
    default_emb = 1 if (is_p1 and "sim" not in config.get("branch_names", [])) else 0
    emb_id = default_emb if args.embodiment_id is None else args.embodiment_id
    cases = []
    sources = []
    for index, (scenario, task, layout, target) in enumerate(TASKS[:1] if args.button_comparison else TASKS):
        path = source / task / layout / "scene_info.json"
        scene = json.loads(path.read_text())
        sources.append({"scenario": scenario, "genie_task": task, "instance": layout, "sha256": sha256(path)})
        cfg = morphmimic_env_cfg("q1", subset_yaml=str(run / "reference/subset.yaml"), num_envs=1, play=True, eval=True,
                                  goal_anchor=frozen_mimic["goal_anchor"], ref_feedback=True)
        cfg.scene.spec_fn = scene_callback(scene, run / "assets", target)
        env = CpuPreviewEnv(cfg, device="cpu")
        try:
            jobs = [("hold", args.seeds[0]), *(("wam_mimic", seed) for seed in args.seeds)]
            if args.button_comparison:
                jobs.insert(0, ("reference", args.seeds[0]))
            for mode, seed in jobs:
                replay = out / f"{scenario}-{f'seed{seed}' if mode == 'wam_mimic' else mode}.mp4"
                targets = button_reference(env, rests, args.steps, mimic_data / "q1/walk1_subject5.npz") if mode == "reference" else None
                result = rollout(env, policy, meta, wam, vae, bank[f"task_{index}"][None], act_stats,
                                 rests, optical, scenario, seed, replay, args.steps, mode, scripted=targets, state_stats=st_stats, emb_id=emb_id)
                cases.append(result)
                print(json.dumps({k: v for k, v in result.items() if k not in ("planning_calls", "physics")}), flush=True)
        finally:
            env.close()
    write_json(out / "manifest.json", {"kind": "geniesim_q1", "step": 5000, "mimic_step": 12000,
              "wam_checkpoint_sha256": sha256(args.wam_checkpoint), "mimic_checkpoint_sha256": sha256(fp),
              "scenes": sources, "asset_revision": json.loads((run / "genie-assets.json").read_text())["revision"],
              "genie_commit": subprocess.check_output(["git", "-c", f"safe.directory={run / 'genie_sim'}", "-C",
                                                        str(run / "genie_sim"), "rev-parse", "HEAD"], text=True).strip(),
              "backend": "MuJoCo CPU", "robot": "Q1", "optical_calibration": None if optical is None else optical[-1], "cases": cases})


if __name__ == "__main__":
    main()
