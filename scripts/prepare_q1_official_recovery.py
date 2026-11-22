"""Validate the pinned Primebot Q1 get-up and adapt its collision floor."""

import argparse
import json
from pathlib import Path
import shutil

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from scripts.prepare_mimic_robot_five import reference_five, sha, save_json

REVISION = '2153a6c561b3b755ae5bcca89683cb66d5ad5852'
SOURCE_SHA = '133e79d897eef3d7a2988019245f75b460e5cfac2d7571b4447345aa71db4344'


def lowest_collision(model, data):
    lowest = float('inf')
    for g in range(model.ngeom):
        if not (model.geom_contype[g] or model.geom_conaffinity[g]):
            continue
        z = data.geom_xmat[g].reshape(3, 3)[2]
        size = model.geom_size[g]
        kind = model.geom_type[g]
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[g]
            a = model.mesh_vertadr[mesh]
            support = -(model.mesh_vert[a:a + model.mesh_vertnum[mesh]] @ z).min()
        elif kind == mujoco.mjtGeom.mjGEOM_BOX:
            support = abs(z) @ size
        elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            support = size[0]
        elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            support = size[0] + abs(z[2]) * size[1]
        elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            support = size[0] * np.linalg.norm(z[:2]) + abs(z[2]) * size[1]
        elif kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            support = np.linalg.norm(z * size)
        else:
            raise ValueError('Unsupported native collision geometry')
        lowest = min(lowest, data.geom_xpos[g, 2] - support)
    return lowest


def prepare(source, output):
    from morphmimic.constants import HUMAN_JOINTS, HUMAN_YAW_OFFSET, HUMAN_ROLL_OFFSET
    from morphmimic.robots import get_robot
    from morphmimic.robots.q1 import asset_root

    if sha(source) != SOURCE_SHA:
        raise ValueError('Official Q1 source SHA differs')
    if output.exists():
        raise FileExistsError(output)
    spec = get_robot('q1')
    model = mujoco.MjModel.from_xml_path(str(asset_root() / 'robot/q1.xml'))
    joints = [model.joint(i).name for i in range(1, model.njnt)]
    if tuple(joints) != spec.joint_names:
        raise ValueError('Native joint order differs from official recording')
    names = [model.body(i).name for i in range(1, model.nbody)]
    with np.load(source, allow_pickle=False) as archive:
        raw = {k: archive[k].copy() for k in archive.files}
    if float(raw['fps'].item()) != 50 or raw['joint_pos'].shape != (350, 22):
        raise ValueError('Official frame/clock contract differs')
    data = mujoco.MjData(model)
    maximum = rotation_error = 0.
    floor = []
    for frame in range(350):
        data.qpos[:3] = raw['body_pos_w'][frame, 0]
        data.qpos[3:7] = raw['body_quat_w'][frame, 0]
        data.qpos[7:] = raw['joint_pos'][frame]
        mujoco.mj_forward(model, data)
        maximum = max(maximum, float(abs(data.xpos[1:] - raw['body_pos_w'][frame]).max()))
        dot = (data.xquat[1:] * raw['body_quat_w'][frame]).sum(-1)
        rotation_error = max(rotation_error, float(abs(abs(dot) - 1).max()))
        floor.append(lowest_collision(model, data))
    if maximum > 1e-5 or rotation_error > 1e-5:
        raise ValueError('Official recording fails native FK validation')
    lift = np.maximum(0., .002 - np.asarray(floor))
    indices = [names.index(n) for n in spec.tracked_bodies]
    motion = dict(fps=np.array(50), robot=np.array('q1'), root_body=np.array(spec.root_body),
                  joint_names=np.array(joints), body_names=np.array(spec.tracked_bodies))
    for key in ('joint_pos', 'joint_vel', 'body_pos_w', 'body_quat_w', 'body_lin_vel_w', 'body_ang_vel_w'):
        value = raw[key].copy()
        if key.startswith('body_'):
            value = value[:, indices]
        if key == 'body_pos_w':
            value[..., 2] += lift[:, None]
        if key == 'body_lin_vel_w':
            value[..., 2] += np.gradient(lift, .02)[:, None]
        hold = np.repeat(value[-1:], 150, axis=0)
        if key in ('joint_vel', 'body_lin_vel_w', 'body_ang_vel_w'):
            hold.fill(0)
        motion[key] = np.concatenate((value, hold))
    motion['floor_lift_m'] = np.r_[lift, np.repeat(lift[-1], 150)]
    rest = Rotation.from_euler('z', HUMAN_YAW_OFFSET) * Rotation.from_euler('x', HUMAN_ROLL_OFFSET)
    side, error = reference_five(motion, [spec.tracked_bodies[i] for i in (0, 3, 6, 10, 13)], HUMAN_JOINTS, rest)
    for folder in ('q1', 'robot_reference50/q1', 'human50', 'raw'):
        (output / folder).mkdir(parents=True, exist_ok=True)
    target = output / 'q1/official_q1_getup.npz'
    point = output / 'robot_reference50/q1/official_q1_getup.npz'
    np.savez_compressed(target, **motion)
    np.savez_compressed(point, **side)
    shutil.copyfile(source, output / 'raw/q1_getUp.npz')
    qa = dict(complete=True, native_fk_position_error_m=maximum,
              native_fk_quaternion_dot_error=rotation_error, point_round_trip_rad=error,
              raw_min_collision_z_m=min(floor), initial_floor_lift_m=float(lift[0]),
              maximum_floor_lift_m=float(lift.max()), adapted_min_collision_z_m=float(np.min(np.asarray(floor)+lift)),
              original_joint_positions_preserved=bool(np.array_equal(motion['joint_pos'][:350], raw['joint_pos'])),
              source_sha256=SOURCE_SHA, output_sha256=sha(target), five_point_sha256=sha(point),
              source_frames=350, standing_hold_frames=150, source_fps=50,
              floor_adaptation='Per-frame vertical translation only, 2 mm native collision clearance; preserves official joint poses.',
              policy_success=None)
    save_json(output / 'official-validation.json', qa)
    row = dict(robot='q1', name='official_q1_getup', dataset='primebot_official', split='train',
               source_id='PrimebotTech/primebot_rl_mjlab/' + REVISION + '/q1_getUp.npz',
               output=str(target), output_sha256=sha(target), frames=500,
               human_available=False, five_point_available=True, five_point_source='robot_reference',
               five_point_sidecar=str(point), five_point_sha256=sha(point),
               five_point_provenance=dict(reference_sha256=sha(target)),
               official_motion=dict(repository='PrimebotTech/primebot_rl_mjlab', revision=REVISION,
                                    path='raw/q1_getUp.npz', sha256=SOURCE_SHA,
                                    validation='official-validation.json', validation_sha256=sha(output/'official-validation.json')))
    save_json(output / 'manifest.json', dict(complete=True, official_q1_recovery=True, human_data=False,
              human_root=str(output/'human50'), robot_reference_root=str(output/'robot_reference50'), clips=[row]))
    print(json.dumps(qa), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source.resolve(strict=True), args.output.absolute())
