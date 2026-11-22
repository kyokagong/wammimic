"""Collect measured Q1 button demonstrations with the existing reference expert.

Roots come from SIM_ASSET_ROOT, MIMIC_RUN_ROOT, RAW_DATA_ROOT. This first
collector covers press_button/stub only; other tasks require their own expert.
"""
import argparse
import json
import os
from pathlib import Path

from concurrent.futures import ProcessPoolExecutor, as_completed
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp
import torch

from scripts.evaluate_geniesim_wam_mimic import (
    TASKS, SLOTS, GOAL_K, PROPRIO_DIM, COMMAND_NAME, DT, SPAWN, CpuPreviewEnv, ObsHistory,
    actor_targets, load_policy, matrices, morphmimic_env_cfg, reference,
    scene_callback, split_state,
)
from wammimic.data_pipeline.common import sha256, write_json

BUTTON_INSTRUCTIONS = [
    "Press the {color} button on the table.", "Walk to the table and press the {color} button.",
    "Reach forward and push down the {color} button.", "Please activate the {color} button.",
    "Move closer and press the {color} button with your right hand.",
    "Use your right hand to push the {color} button.", "Approach the table, then press the {color} button.",
    "Push down on the {color} button on the tabletop.", "Go to the {color} button and press it.",
    "Raise your right hand and press the {color} button.", "Press down the {color} button in front of you.",
    "Touch and depress the {color} button.", "Move your right hand onto the {color} button and push.",
    "Step towards the table and activate the {color} button.", "Make the {color} button go down using your right hand.",
]
BLOCK_INSTRUCTIONS = [
    "Touch the {color} block on the table.", "Walk to the table and touch the {color} block.",
    "Reach forward and tap the {color} block with your right hand.", "Locate the {color} block and touch it.",
    "Step closer and place your right hand on the {color} block.", "Use your right hand to touch the {color} block.",
    "Approach the tabletop and touch the {color} block.", "Move your right hand onto the {color} block.",
    "Raise your right hand and tap the {color} block in front of you.", "Reach out and make contact with the {color} block.",
    "Go to the table and touch the matching {color} block.", "Touch the top of the {color} block with your right hand.",
    "Step forward and bring your right hand down onto the {color} block.", "Select and touch the {color} block on the table.",
    "Walk up to the desk and tap the {color} block.",
]
AVOID_BUTTON_INSTRUCTIONS = [
    "Walk around the obstacle on the floor and press the {color} button on the table.",
    "Avoid the floor obstacle, approach the table, and press the {color} button.",
    "Detour around the barrier on the floor, then push down the {color} button.",
    "Step around the obstacle in front of you and activate the {color} button.",
    "Go around the floor obstacle to reach the table and press the {color} button with your right hand.",
    "Bypass the obstacle on the ground, walk up to the table, and press the {color} button.",
    "Navigate around the floor box and push the {color} button on the tabletop.",
    "Walk past the obstacle without touching it, then press the {color} button.",
    "Steer around the obstacle on the floor and depress the {color} button.",
    "Curve around the barrier ahead, approach the desk, and press the {color} button.",
]


def visual_avoid_button_reference(env, rests, steps, walk_clip, detour_side, detour_y, press_depth=.034):
    """Real monotonic curved obstacle-avoidance walk from walk1_subject5 (f=6290..6365) followed by heading-locked button press."""
    model = env.sim.mj_model
    data = mujoco.MjData(model)
    data.qpos[:] = env.reference_qpos(0)
    mujoco.mj_forward(model, data)
    wrist = model.body("robot/right_wrist_yaw_link").id
    hand = model.geom("robot/right_wrist_yaw_link_0_collision").id
    target_geom = model.geom("task_target").id
    mesh = int(model.geom_dataid[target_geom])
    start_v, count_v = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
    vertices = model.mesh_vert[start_v:start_v + count_v] @ data.geom_xmat[target_geom].reshape(3, 3).T
    top = float(data.geom_xpos[target_geom, 2] + vertices[:, 2].max())
    target_xy = data.geom_xpos[target_geom, :2].copy()
    # Compute exact arrival offset from compiled MuJoCo button XY so the robot stops squarely in front of the button.
    advance_x = float(target_xy[0] - data.qpos[0] - 0.175)
    advance_y = float(target_xy[1] - data.qpos[1] + 0.08)
    stop_x = float(data.qpos[0] + advance_x)
    stop_y = float(data.qpos[1] + advance_y)
    cam_yaw = float(np.clip(np.arctan2((target_xy[1] - stop_y) - 0.08, max(target_xy[0] - stop_x, 0.18)), -0.34, 0.05))
    arm = [model.joint("robot/right_" + name + "_joint").id for name in
           ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")]
    address = model.jnt_qposadr[arm]
    neutral_arm = data.qpos[address].copy()
    low, high = env.native_joint_limits.numpy()[[list(env.joint_qadr).index(i) for i in address]].T
    data.qpos[0] = stop_x
    data.qpos[1] = stop_y
    q_yaw = Rotation.from_euler("z", cam_yaw).as_quat()
    data.qpos[3:7] = np.array([q_yaw[3], q_yaw[0], q_yaw[1], q_yaw[2]])
    arm_seed = np.clip(neutral_arm, low + 1e-6, high - 1e-6)
    waypoints = (
        np.r_[target_xy + np.array([-0.04, 0.0]), top + 0.085],
        np.r_[target_xy, top + 0.015],
        np.r_[target_xy, top - press_depth],
    )
    positions, rotations = [], []
    for target in waypoints:
        def error(q):
            data.qpos[address] = q
            mujoco.mj_forward(model, data)
            geom_rotation = data.geom_xmat[hand].reshape(3, 3)
            tip = data.geom_xpos[hand].copy()
            tip[2] -= np.abs(geom_rotation[2]) @ model.geom_size[hand]
            return np.r_[tip - target, .006 * (q - arm_seed)]

        trial = least_squares(error, arm_seed, bounds=(low, high), max_nfev=120)
        gap = np.linalg.norm(error(trial.x)[:3])
        if gap > .03:
            raise ValueError(f"Avoid-button target outside arm reach: residual {gap:.3f} m")
        arm_seed = trial.x
        positions.append(data.xpos[wrist].copy())
        rotations.append(data.xmat[wrist].reshape(3, 3).copy())

    def smooth(t, t0, t1):
        u = np.clip((t - t0) / (t1 - t0), 0., 1.)
        return u**3 * (10. + u * (-15. + 6. * u))

    xyz, orientation, human_xyz, human_orientation = rests
    t = np.arange(steps + GOAL_K + 1) * DT
    robot = np.repeat(xyz[None], len(t), axis=0)
    robot_rot = np.repeat(orientation[None], len(t), axis=0)

    with np.load(walk_clip) as walk:
        source = walk["body_pos_w"][:, list(SLOTS[:3])].astype(float)
        source_rot = matrices(walk["body_quat_w"][:, list(SLOTS[:3])])
    # Use clean start-from-rest walk f=238..365 rotated toward the detour heading (~±28 deg) so Mimic tracks with <9 cm error.
    start, finish = 238, 365
    walk_steps = 115
    walk_end_t = walk_steps * DT
    timing = np.arange(len(t)) * ((finish - start) / float(walk_steps))
    frame = np.clip(np.rint(start + timing).astype(int), start, finish)

    walk_dist = float(np.hypot(advance_x, advance_y))
    walk_theta = float(np.arctan2(advance_y, max(advance_x, 1e-3)))
    start_yaw = float(np.arctan2(source_rot[start, 0, 1, 0], source_rot[start, 0, 0, 0]))
    align_x = Rotation.from_euler("z", -start_yaw).as_matrix()

    delta_local = np.einsum("ij,tpj->tpi", align_x, source[frame] - source[start, :1])
    delta_local -= delta_local[0:1]
    src_dx = max(float(delta_local[-1, 0, 0]), 1e-3)
    delta_local[..., 0] *= walk_dist / src_dx
    delta_local[..., 1] *= 0.25

    rot_theta = Rotation.from_euler("z", walk_theta).as_matrix()
    delta_world = np.einsum("ij,tpj->tpi", rot_theta, delta_local)
    robot[:, :3] += delta_world

    pelvis_rot = source_rot[frame, 0]
    walk_yaw = np.arctan2(pelvis_rot[:, 1, 0], pelvis_rot[:, 0, 0])
    arrive_blend = smooth(t, walk_end_t - 0.50, walk_end_t + 0.35)
    target_yaw = walk_theta + (cam_yaw - walk_theta) * arrive_blend
    unyaw = Rotation.from_euler("z", (target_yaw - walk_yaw)[:, None]).as_matrix()
    robot_rot[:, :3] = unyaw[:, None] @ source_rot[frame]
    robot_rot[:, 0] = Rotation.from_euler("z", target_yaw[:, None]).as_matrix() @ orientation[0]

    # Carry wrists with the pelvis along the detour, then lift right wrist into wam_head FOV and press button.
    yaw_mats = Rotation.from_euler("z", target_yaw[:, None]).as_matrix()
    wrist_rest_rel = xyz[3:] - xyz[:1]
    robot[:, 3:] = robot[:, :1] + np.einsum("tij,pj->tpi", yaw_mats, wrist_rest_rel)
    robot_rot[:, 3:] = yaw_mats[:, None] @ orientation[None, 3:]

    # Early pre-lift brings the working wrist into wam_head FOV during detour and presses firmly through contact.
    approach = smooth(t, 0.35, 1.35)
    second = smooth(t, walk_end_t + 0.05, walk_end_t + 0.70)
    third = smooth(t, walk_end_t + 0.70, walk_end_t + 1.45)
    home_rw = robot[-1, 4].copy()
    robot[:, 4] += (approach[:, None] * (positions[0] - home_rw)
                    + second[:, None] * (positions[1] - positions[0])
                    + third[:, None] * (positions[2] - positions[1]))
    end_rw_rot = robot_rot[-1, 4].copy()
    quaternion_path = Rotation.from_matrix(np.stack([end_rw_rot, *rotations]))
    robot_rot[:, 4] = Slerp(np.arange(4), quaternion_path)(approach + second + third).as_matrix()

    human = (human_xyz + SPAWN)[None] + (robot - xyz[None]) * 1.65
    human_rot = robot_rot @ orientation.swapaxes(-1, -2) @ human_orientation
    return (human, human_rot), (robot, robot_rot)


def visual_tabletop_reference(env, rests, steps, walk_clip, advance=.18, press_depth=.034, *, touch_only=False):
    """Heading-locked approach with early wrist pre-lift so target and working wrist stay in wam_head FOV."""
    model = env.sim.mj_model
    data = mujoco.MjData(model)
    data.qpos[:] = env.reference_qpos(0)
    mujoco.mj_forward(model, data)
    wrist = model.body("robot/right_wrist_yaw_link").id
    hand = model.geom("robot/right_wrist_yaw_link_0_collision").id
    target_geom = model.geom("task_target").id
    mesh = int(model.geom_dataid[target_geom])
    if mesh >= 0:
        start_v, count_v = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        vertices = model.mesh_vert[start_v:start_v + count_v] @ data.geom_xmat[target_geom].reshape(3, 3).T
        top = float(data.geom_xpos[target_geom, 2] + vertices[:, 2].max())
    else:
        top = float(data.geom_xpos[target_geom, 2] + model.geom_size[target_geom, 2])
    target_xy = data.geom_xpos[target_geom, :2].copy()
    arm = [model.joint("robot/right_" + name + "_joint").id for name in
           ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")]
    address = model.jnt_qposadr[arm]
    neutral_arm = data.qpos[address].copy()
    low, high = env.native_joint_limits.numpy()[[list(env.joint_qadr).index(i) for i in address]].T
    data.qpos[0] += advance
    arm_seed = np.clip(neutral_arm, low + 1e-6, high - 1e-6)
    waypoints = (
        (np.r_[target_xy + np.array([-0.04, 0.0]), top + 0.085],
         np.r_[target_xy, top + 0.015],
         np.r_[target_xy, top - (0.012 if touch_only else press_depth)])
    )
    positions, rotations = [], []
    for target in waypoints:
        def error(q):
            data.qpos[address] = q
            mujoco.mj_forward(model, data)
            geom_rotation = data.geom_xmat[hand].reshape(3, 3)
            tip = data.geom_xpos[hand].copy()
            tip[2] -= np.abs(geom_rotation[2]) @ model.geom_size[hand]
            return np.r_[tip - target, .006 * (q - arm_seed)]

        trial = least_squares(error, arm_seed, bounds=(low, high), max_nfev=120)
        gap = np.linalg.norm(error(trial.x)[:3])
        if gap > .03:
            raise ValueError(f"Target outside arm reach: residual {gap:.3f} m")
        arm_seed = trial.x
        positions.append(data.xpos[wrist].copy())
        rotations.append(data.xmat[wrist].reshape(3, 3).copy())

    def smooth(t, t0, t1):
        u = np.clip((t - t0) / (t1 - t0), 0., 1.)
        return u**3 * (10. + u * (-15. + 6. * u))

    xyz, orientation, human_xyz, human_orientation = rests
    t = np.arange(steps + GOAL_K + 1) * DT
    robot = np.repeat(xyz[None], len(t), axis=0)
    robot_rot = np.repeat(orientation[None], len(t), axis=0)
    with np.load(walk_clip) as walk:
        source = walk["body_pos_w"][:, list(SLOTS[:3])].astype(float)
        source_rot = matrices(walk["body_quat_w"][:, list(SLOTS[:3])])
    start, finish = 238, 365
    timing = np.arange(len(t)) * ((finish - start) / 105)
    frame = np.clip(np.rint(start + timing).astype(int), start, finish)
    # Compute heading toward midpoint of target and right-wrist approach so both target and wrist sit centered in wam_head FOV.
    cam_yaw = float(np.clip(np.arctan2(target_xy[1] - 0.11, max(target_xy[0] - advance, 0.20)), -0.36, 0.05))
    # Keep pelvis yaw aimed at target/wrist midpoint while preserving subtle natural stepping.
    pelvis_rot = source_rot[frame, 0]
    walk_yaw = np.arctan2(pelvis_rot[:, 1, 0], pelvis_rot[:, 0, 0])
    unyaw = Rotation.from_euler("z", (cam_yaw - walk_yaw)[:, None]).as_matrix()
    shift = source[frame] - source[start]
    shift[..., 0] *= advance / max(source[finish, 0, 0] - source[start, 0, 0], 1e-3)
    shift[..., 1] *= .20
    robot[:, :3] += shift
    robot_rot[:, :3] = unyaw[:, None] @ source_rot[frame]
    robot_rot[:, 0] = Rotation.from_euler("z", cam_yaw).as_matrix() @ orientation[0]
    robot[:, 3:] += shift[:, :1]

    # Early pre-lift brings the working wrist into wam_head FOV during approach and keeps it framed through contact.
    approach = smooth(t, .08, .75)
    second = smooth(t, 1.25, 1.85)
    third = smooth(t, 1.85, 2.55)
    home = xyz[4] + shift[-1, 0]
    robot[:, 4] += (approach[:, None] * (positions[0] - home)
                    + second[:, None] * (positions[1] - positions[0])
                    + third[:, None] * (positions[2] - positions[1]))
    quaternion_path = Rotation.from_matrix(np.stack([orientation[4], *rotations]))
    robot_rot[:, 4] = Slerp(np.arange(4), quaternion_path)(approach + second + third).as_matrix()
    human = (human_xyz + SPAWN)[None] + (robot - xyz[None]) * 1.65
    human_rot = robot_rot @ orientation.swapaxes(-1, -2) @ human_orientation
    return (human, human_rot), (robot, robot_rot)


@torch.inference_mode()
def collect(seed, output, assets, mimic, policy, meta, frozen, mimic_data, rests, steps, task_mode="press_button"):
    rng = np.random.default_rng(seed)
    avoid_mode = (task_mode == "avoid_obstacle_then_press_button")
    task_idx = 1 if task_mode == "matching_block_contact" else 0
    _, task, layout, target_name = TASKS[task_idx]
    source = assets / "genie_sim/source/geniesim_scene_agent/examples/tasks" / task / layout / "scene_info.json"
    scene = json.loads(source.read_text())
    if avoid_mode:
        table_advance = float(rng.uniform(-0.02, 0.05))
        detour_side = 1 if (seed % 2 == 0) else -1
        detour_y = 0.20
        # Keep table X within Q1's natural 1-stride reach (~0.18-0.22m X) while shifting laterally (±0.20-0.24m Y) around the obstacle.
        shift = np.array([table_advance, detour_side * float(rng.uniform(0.32, 0.40)), float(rng.uniform(-.01, .01))])
        # Place bright floor obstacle ahead of spawn on the inside straight path with clean foot clearance.
        obs_x = float(-0.43 + rng.uniform(0.22, 0.28))
        obs_y = float(-detour_side * rng.uniform(0.09, 0.13))
        obs_half = [float(rng.uniform(0.038, 0.050)), float(rng.uniform(0.038, 0.050)), float(rng.uniform(0.14, 0.19))]
        obs_rgba = [*rng.choice([[0.95, 0.42, 0.08], [0.92, 0.78, 0.12], [0.85, 0.18, 0.18], [0.16, 0.72, 0.28]]), 1.0]
    else:
        table_advance = 0.0
        shift = rng.uniform([-.03, -.025, -.015], [.025, .025, .015])
        detour_side, detour_y = 0, 0.0
        obs_x = obs_y = 0.0
        obs_half = obs_rgba = None
    color = str(rng.choice(["blue", "red", "purple", "green", "yellow"]))
    for key, row in scene["layout"].items():
        row["xyz"] = (np.asarray(row["xyz"]) + shift).tolist()
        if key == target_name:
            row["description"]["color"] = color
    base = scene_callback(scene, assets / "assets", target_name)
    pitch = float(rng.uniform(33, 38)) if avoid_mode else float(rng.uniform(24, 29))

    def configure(spec):
        base(spec)
        angle = np.deg2rad(pitch)
        spec.camera("wam_head").alt.xyaxes = [0., -1., 0., np.sin(angle), 0., np.cos(angle)]
        if avoid_mode:
            body = spec.worldbody.add_body(name="floor_obstacle_body", pos=[obs_x, obs_y, obs_half[2]])
            body.add_geom(name="floor_obstacle", type=mujoco.mjtGeom.mjGEOM_BOX, size=obs_half, rgba=obs_rgba, contype=1, conaffinity=1)
        for geom in spec.geoms:
            if "floor" in geom.name or "terrain" in geom.name or "table_000" in geom.name:
                if geom.name != "floor_obstacle":
                    geom.rgba = [*rng.uniform(.18, .8, 3), 1.]
        for light in spec.lights:
            light.diffuse = rng.uniform(.45, .95, 3)

    cfg = morphmimic_env_cfg("q1", subset_yaml=str(output / "reference/subset.yaml"), num_envs=1,
                             play=True, eval=True, goal_anchor=frozen["goal_anchor"], ref_feedback=True)
    cfg.scene.spec_fn = configure
    env = CpuPreviewEnv(cfg, device="cpu")
    try:
        touch_only = (task_mode == "matching_block_contact")
        if avoid_mode:
            goals = visual_avoid_button_reference(
                env, rests, steps + 50, mimic_data / "q1/walk1_subject5.npz",
                detour_side=detour_side, detour_y=detour_y,
            )
        else:
            goals = visual_tabletop_reference(
                env, rests, steps + 50, mimic_data / "q1/walk1_subject5.npz",
                advance=float(rng.uniform(.15, .21)), touch_only=touch_only,
            )
        ids = torch.tensor([0])
        env.command_manager.get_term(COMMAND_NAME).assign_clips(ids, ids)
        obs, _ = env.reset(env_ids=ids)
        history = ObsHistory(1, meta.hist_len, PROPRIO_DIM, env.device)
        history.reset(ids, obs["policy"])
        model, data = env.sim.mj_model, env.native_data
        target = model.geom("task_target").id
        obstacle_id = model.geom("floor_obstacle").id if avoid_mode else -1
        robot_geoms = {i for i in range(model.ngeom) if "robot/" in model.body(model.geom_bodyid[i]).name}
        button_qadr = None if touch_only else model.jnt_qposadr[model.joint("button_travel").id]
        hand_geoms = [{i for i in range(model.ngeom) if f"/{side}_wrist" in model.geom(i).name}
                      for side in ("left", "right")]
        rows, images, image_indices, touches, joints, velocities, commands = [], [], [], [], [], [], []
        projections, camera_positions, camera_rotations = [], [], []
        cam = model.camera("wam_head").id
        success, reason = False, "timeout"
        obstacle_collisions = 0
        max_lat_y = 0.0
        min_obs_dist = float("inf")
        button_max_travel = 0.0
        ref_k = 0
        with mujoco.Renderer(model, height=224, width=224) as renderer:
            for k in range(steps):
                state = env.state()
                parts = split_state(state)
                act_p0 = parts.body_pos[0, list(SLOTS)[0]].cpu().numpy()
                max_lat_y = max(max_lat_y, float(abs(act_p0[1])))
                if avoid_mode:
                    min_obs_dist = min(min_obs_dist, float(np.linalg.norm(act_p0[:2] - np.array([obs_x, obs_y]))))
                force = np.zeros(2)
                wrench = np.zeros(6)
                for i, contact in enumerate(data.contact[:data.ncon]):
                    a, b = map(int, contact.geom)
                    if avoid_mode and ((a == obstacle_id and b in robot_geoms) or (b == obstacle_id and a in robot_geoms)):
                        if contact.dist <= 0.001:
                            obstacle_collisions += 1
                    for side, geoms in enumerate(hand_geoms):
                        if (a == target and b in geoms) or (b == target and a in geoms):
                            mujoco.mj_contactForce(model, data, i, wrench)
                            force[side] += max(0., float(wrench[0]))
                touches.append(force)
                if touch_only:
                    success |= bool(force.max() >= .5)
                else:
                    travel = float(data.qpos[button_qadr])
                    button_max_travel = max(button_max_travel, travel)
                    success |= bool(travel >= .003 and force.max() >= .5)
                rows.append(state[0].cpu().numpy().copy())
                joints.append(data.qpos[env.joint_qadr].copy())
                velocities.append(data.qvel[env.joint_dadr].copy())
                if k % 5 == 0:
                    points = np.stack((data.geom_xpos[target], parts.body_pos[0, list(SLOTS)[-1]].cpu().numpy()))
                    local = (points - data.cam_xpos[cam]) @ data.cam_xmat[cam].reshape(3, 3)
                    depth = -local[:, 2]
                    half = np.tan(np.deg2rad(model.cam_fovy[cam]) / 2)
                    uv = np.c_[.5 + local[:, 0] / (2 * half * depth), .5 - local[:, 1] / (2 * half * depth)]
                    projections.append(np.c_[uv, depth])
                    camera_positions.append(data.cam_xpos[cam].copy())
                    camera_rotations.append(data.cam_xmat[cam].reshape(3, 3).copy())
                    renderer.update_scene(data, camera="wam_head")
                    images.append(renderer.render().copy())
                    image_indices.append(k)
                action = policy(history.flat(), actor_targets(state, *goals, k))
                if not torch.isfinite(action).all():
                    raise ValueError("Nonfinite expert action")
                commands.append((env.default_joint_pos + env.spec.translator.to_native_delta(
                    action * env.joint_mask * env.action_scale))[0].cpu().numpy().copy())
                obs, _, ended, truncated, _ = env.step(action)
                history.push(obs["policy"])
                if bool(ended[0]) or bool(truncated[0]):
                    reason = "termination"
                    break
        diagnostics = env.physics_diagnostics.report()
        projected = np.stack(projections)
        visible = ((projected[..., :2] >= .03) & (projected[..., :2] <= .97)).all(-1) & (projected[..., 2] > 0)
        target_fraction = float(visible[:, 0].mean())
        wrist_fraction = float((visible[:, 0] & visible[:, 1]).mean())
        if avoid_mode:
            accepted = (success and reason != "termination" and diagnostics["zero_overflow_observed"]
                        and obstacle_collisions == 0 and max_lat_y >= 0.14
                        and target_fraction >= .60 and wrist_fraction >= .20)
            prompts = AVOID_BUTTON_INSTRUCTIONS
        else:
            accepted = success and reason != "termination" and diagnostics["zero_overflow_observed"] and target_fraction >= .8 and wrist_fraction >= .4
            prompts = BUTTON_INSTRUCTIONS if task_mode == "press_button" else BLOCK_INSTRUCTIONS
        report = {"seed": seed, "accepted": accepted, "physical_button_success": success,
                  "button_max_travel_m": round(button_max_travel, 5) if button_qadr is not None else None,
                  "obstacle_collision_steps": obstacle_collisions, "max_lateral_detour_m": round(max_lat_y, 4),
                  "min_obstacle_pelvis_dist_m": round(min_obs_dist, 4) if avoid_mode and np.isfinite(min_obs_dist) else None,
                  "termination": reason, "task": task_mode, "robot": "q1", "embodiment_tag": "bimanual_stub",
                  "label_provenance": "mujoco_simulated_robot_state", "contact_threshold_n": .5,
                  "expert": "curved_avoid_obstacle_button_ik + Mimic 12000" if avoid_mode else "heading_locked_prelift_tabletop_ik + Mimic 12000; visual gate enforced",
                  "scene_source_sha256": sha256(source), "scene": scene, "camera_pitch_degrees": pitch,
                  "obstacle": {"pos": [obs_x, obs_y, obs_half[2]], "half_size": obs_half, "detour_side": detour_side} if avoid_mode else None,
                  "camera_fps": 10, "physics_state_fps": 50, "frames": len(rows),
                  "instruction": prompts[seed % len(prompts)].format(color=color),
                  "target_in_frustum_fraction": target_fraction, "target_and_wrist_in_frustum_fraction": wrist_fraction,
                  "split": "test" if seed % 10 == 0 else "val" if seed % 10 == 1 else "train",
                  "physics": diagnostics, "joint_names_native": list(env.spec.joint_names),
                  "limits": "Q1 tabletop obstacle-avoidance expert; bimanual_stub contact labels; no finger joints"}
        if success:
            folder = output / ("episodes" if accepted else "rejected_visual") / f"seed_{seed:06d}"
            folder.mkdir(parents=True, exist_ok=True)
            states = np.stack(rows)
            parts = split_state(torch.from_numpy(states))
            poses = np.concatenate((parts.body_pos[:, list(SLOTS)].numpy(), parts.body_quat[:, list(SLOTS)].numpy()), axis=-1)
            assert np.isfinite(states).all() and np.isfinite(touches).all() and poses.shape[1:] == (5, 7)
            np.savez_compressed(folder / "trajectory.npz", time_s=np.arange(len(rows)) / 50,
                states_canonical=states, poses_world=poses, joint_pos=np.stack(joints),
                joint_vel=np.stack(velocities), joint_pos_target=np.stack(commands), rgb=np.stack(images), rgb_frame_index=np.array(image_indices),
                camera_position_w=np.stack(camera_positions), camera_rotation_w=np.stack(camera_rotations), target_wrist_uv_depth=projected,
                touch_force_n=np.stack(touches), touch=(np.stack(touches) >= .5).astype(np.float32))
            report["trajectory_sha256"] = sha256(folder / "trajectory.npz")
            write_json(folder / "episode.json", report)
        return report
    finally:
        env.close()


_WORKER_STATE = {}


def _init_worker(assets_str, mimic_str, output_str, rests):
    torch.set_num_threads(1)
    assets, mimic, output = Path(assets_str), Path(mimic_str), Path(output_str)
    frozen = json.loads((mimic / "config.json").read_text())
    mimic_data = Path(frozen["prepared_manifest"]).parent
    checkpoint = mimic / "q1/ckpt_12000.pt"
    policy, meta = load_policy(checkpoint, "cpu", interface="human", human_points="five", kp_horizon=GOAL_K,
                               goal_anchor=frozen["goal_anchor"])
    _WORKER_STATE.update(assets=assets, mimic=mimic, output=output, policy=policy, meta=meta,
                         frozen=frozen, mimic_data=mimic_data, rests=rests, ckpt_sha=sha256(checkpoint))


def _run_seed(args_tuple):
    seed, task_mode = args_tuple
    st = _WORKER_STATE
    receipt = st["output"] / "attempts" / f"seed_{seed:06d}.json"
    if receipt.exists():
        return json.loads(receipt.read_text())
    try:
        ep_steps = 225 if task_mode == "avoid_obstacle_then_press_button" else 160
        report = collect(seed, st["output"], st["assets"], st["mimic"], st["policy"], st["meta"],
                         st["frozen"], st["mimic_data"], st["rests"], ep_steps, task_mode=task_mode)
    except ValueError as error:
        report = {"seed": seed, "task": task_mode, "accepted": False, "error": str(error)}
    report["mimic_checkpoint_sha256"] = st["ckpt_sha"]
    write_json(receipt, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--successes", type=int, default=0, help="Stop after this many accepted episodes; zero uses the attempt limit")
    parser.add_argument("--task", choices=["press_button", "matching_block_contact", "avoid_obstacle_then_press_button"], default="press_button")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.attempts <= 0 or args.first_seed < 0 or args.successes < 0 or args.workers <= 0:
        parser.error("Invalid attempt, seed, success, or worker count")
    assets, mimic = Path(os.environ["SIM_ASSET_ROOT"]), Path(os.environ["MIMIC_RUN_ROOT"])
    output = Path(os.environ["RAW_DATA_ROOT"]) / "sim_loco_tabletop"
    output.mkdir(parents=True, exist_ok=True)
    # Prime shared reference cache before spawning workers.
    frozen = json.loads((mimic / "config.json").read_text())
    mimic_data = Path(frozen["prepared_manifest"]).parent
    rests = reference(output, mimic_data, 300)
    accepted = sum(json.loads(p.read_text()).get("accepted", False) for p in (output / "attempts").glob("*.json"))
    seeds = list(range(args.first_seed, args.first_seed + args.attempts))
    if args.workers == 1:
        _init_worker(str(assets), str(mimic), str(output), rests)
        for seed in seeds:
            if args.successes and accepted >= args.successes:
                break
            report = _run_seed((seed, args.task))
            accepted += int(bool(report.get("accepted")))
            print(json.dumps({k: report.get(k) for k in ("seed", "task", "accepted", "physical_button_success",
                                                         "target_in_frustum_fraction", "target_and_wrist_in_frustum_fraction", "error")}), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                                 initargs=(str(assets), str(mimic), str(output), rests)) as pool:
            pending = {}
            seed_iter = iter(seeds)
            while len(pending) < args.workers * 2:
                try:
                    s = next(seed_iter)
                except StopIteration:
                    break
                pending[pool.submit(_run_seed, (s, args.task))] = s
            while pending:
                for fut in as_completed(list(pending)):
                    pending.pop(fut)
                    report = fut.result()
                    accepted += int(bool(report.get("accepted")))
                    print(json.dumps({k: report.get(k) for k in ("seed", "task", "accepted", "physical_button_success",
                                                                 "target_in_frustum_fraction", "target_and_wrist_in_frustum_fraction", "error")}), flush=True)
                    if args.successes and accepted >= args.successes:
                        for other in pending:
                            other.cancel()
                        pending.clear()
                        break
                    try:
                        s = next(seed_iter)
                        pending[pool.submit(_run_seed, (s, args.task))] = s
                    except StopIteration:
                        pass
                    break
    if args.successes and accepted < args.successes:
        raise RuntimeError(f"Only {accepted}/{args.successes} episodes passed physics and visual gates")


if __name__ == "__main__":
    main()
