"""Add provenance-bound MotionDecode reference keypoints to an immutable dataset."""

import argparse
from collections import Counter
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
from scipy.spatial.transform import Rotation

SLOTS = (0, 7, 8, 20, 21)
BODY_SLOTS = (0, 3, 6, 10, 13)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def link_or_copy(source, target):
    try:
        os.link(source, target)
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        shutil.copyfile(source, target)


def reference_five(motion, names, joint_names, rest):
    """Keep metric FK positions and explicitly transport native local axes."""
    body_names = motion['body_names'].tolist()
    indices = [body_names.index(name) for name in names]
    positions = motion['body_pos_w'][:, indices].astype(np.float32)
    native = motion['body_quat_w'][:, indices].astype(np.float64)
    n = len(motion['joint_pos'])
    if (int(motion['fps']) != 50 or positions.shape != (n, 5, 3)
            or native.shape != (n, 5, 4) or not np.isfinite(positions).all()
            or not np.isfinite(native).all()
            or not np.allclose(np.linalg.norm(native, axis=-1), 1, atol=1e-3, rtol=0)):
        raise ValueError('Invalid reference poses, frame count or 50 Hz clock')
    rotation = Rotation.from_quat(native.reshape(-1, 4), scalar_first=True)
    transported = (rotation * rest).as_quat(scalar_first=True).reshape(n, 5, 4).astype(np.float32)
    recovered = Rotation.from_quat(transported.reshape(-1, 4), scalar_first=True) * rest.inv()
    error = float((recovered.inv() * rotation).magnitude().max())
    if error > 1e-6:
        raise ValueError('Robot reference orientation round-trip failed')
    pos = np.zeros((n, 24, 3), np.float32)
    quat = np.zeros((n, 24, 4), np.float32)
    quat[..., 0] = 1
    pos[:, SLOTS] = positions
    quat[:, SLOTS] = transported
    valid = np.zeros(24, bool)
    valid[list(SLOTS)] = True
    return dict(fps=np.array(50), human_pos_w=pos, human_quat_w=quat,
                human_joint_names=np.array(joint_names), human_joint_valid=valid,
                source_time_s=np.arange(n, dtype=np.float64) / 50,
                five_point_source=np.array('robot_reference'), native_quat_w=native.astype(np.float32),
                body_names=np.array(names), native_to_transport_wxyz=rest.as_quat(scalar_first=True),
                adapter_version=np.array('robot-reference-smpl-transport-v1')), error


def prepare(parent, output):
    from morphmimic.constants import HUMAN_JOINTS, HUMAN_YAW_OFFSET, HUMAN_ROLL_OFFSET
    from morphmimic.robots import get_robot

    parent = parent.resolve(strict=True)
    output = output.absolute()
    stage = output.with_name(output.name + '.processing')
    if output.exists() or stage.exists():
        raise FileExistsError('Choose a fresh output directory')
    original = json.loads((parent / 'manifest.json').read_text())
    if original.get('complete') is not True:
        raise ValueError('Parent dataset is incomplete')
    rest = Rotation.from_euler('z', HUMAN_YAW_OFFSET) * Rotation.from_euler('x', HUMAN_ROLL_OFFSET)
    mappings = {r: [get_robot(r).tracked_bodies[i] for i in BODY_SLOTS] for r in ('q1', 'bumi', 'g1')}

    def relocate(value):
        if isinstance(value, dict):
            return {k: relocate(v) for k, v in value.items()}
        if isinstance(value, list):
            return [relocate(v) for v in value]
        if isinstance(value, str) and (value == str(parent) or value.startswith(str(parent) + '/')):
            return str(output) + value[len(str(parent)):]
        return value

    manifest = relocate(original)
    counts = Counter()
    maximum_error = 0.
    try:
        shutil.copytree(parent, stage, copy_function=link_or_copy)
        for row, old in zip(manifest['clips'], original['clips'], strict=True):
            relative = Path(old['output']).relative_to(parent)
            source = stage / relative
            if sha(source) != old['output_sha256']:
                raise ValueError(f'Reference hash differs for {relative}')
            if row['dataset'] == 'motiondecode':
                target = stage / 'robot_reference50' / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with np.load(source, allow_pickle=False) as motion:
                    arrays, error = reference_five(motion, mappings[row['robot']], HUMAN_JOINTS, rest)
                maximum_error = max(maximum_error, error)
                np.savez_compressed(target, **arrays)
                row.update(five_point_source='robot_reference', five_point_available=True,
                           five_point_sidecar=str(output / target.relative_to(stage)), five_point_sha256=sha(target),
                           five_point_provenance=dict(reference_sha256=old['output_sha256'],
                               bodies=mappings[row['robot']], frame_mapping='identity', position_scale=1.,
                               world_frame='FLU', adapter_version='robot-reference-smpl-transport-v1'))
                counts[row['robot'], row['split']] += 1
            else:
                if not row.get('human_available'):
                    raise ValueError('Unexpected unpaired non-MotionDecode clip')
                target = stage / Path(old['human_sidecar']).relative_to(parent)
                if sha(target) != old['human_sha256']:
                    raise ValueError('Original human sidecar hash differs')
                row.update(five_point_source='original_human', five_point_available=True,
                           five_point_sidecar=row['human_sidecar'], five_point_sha256=row['human_sha256'])
            mixed = stage / 'keypoints50' / relative
            mixed.parent.mkdir(parents=True, exist_ok=True)
            os.link(target, mixed)
        for robot in mappings:
            for split in ('train', 'val'):
                subset = relocate(json.loads((parent / robot / (split + '.yaml')).read_text()))
                save_json(stage / robot / (split + '.yaml'), subset)
                expected = {r['name'] for r in manifest['clips'] if r['robot'] == robot and r['split'] == split}
                if {Path(p).stem for p in subset['clips']} != expected:
                    raise ValueError('Subset membership differs from frozen split')
        review = relocate(json.loads((parent / 'standing-review.json').read_text()))
        save_json(stage / 'standing-review.json', review)
        manifest['neutral_review']['sha256'] = sha(stage / 'standing-review.json')
        manifest.update(five_point_root=str(output / 'keypoints50'),
                        robot_reference_root=str(output / 'robot_reference50'),
                        five_point_sources=['original_human', 'robot_reference'],
                        five_point_parent=dict(path=str(parent / 'manifest.json'), sha256=sha(parent / 'manifest.json')))
        save_json(stage / 'manifest.json', manifest)
        digest = sha(stage / 'manifest.json')
        windows = relocate(json.loads((parent / 'windows.json').read_text()))
        windows.update(training_manifest_sha256=digest, reference_manifest_sha256=digest,
                       subset_sha256={r: sha(stage / r / 'train.yaml') for r in mappings})
        windows['approved_target']['review_manifest']['sha256'] = manifest['neutral_review']['sha256']
        save_json(stage / 'windows.json', windows)
        floor = relocate(json.loads((parent / 'contact-floor.json').read_text()))
        floor['prepared_manifest_sha256'] = digest
        save_json(stage / 'contact-floor.json', floor)
        report = dict(complete=True, parent_manifest_sha256=sha(parent / 'manifest.json'),
                      manifest_sha256=digest, max_orientation_round_trip_rad=maximum_error,
                      robot_reference_counts={f'{r}/{s}': n for (r, s), n in counts.items()},
                      original_human_sidecars_unchanged=True, split_membership_preserved=True,
                      position_error_m=0., body_mapping=mappings)
        save_json(stage / 'five-point-validation.json', report)
        os.rename(stage, output)
        print(json.dumps(report), flush=True)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.parent, args.output)
