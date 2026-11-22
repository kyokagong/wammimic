"""Append the validated official Q1 recovery to a fresh mixed-source dataset."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil

from scripts.prepare_mimic_robot_five import link_or_copy, save_json, sha
from scripts.prepare_q1_official_recovery import SOURCE_SHA


def append(parent, official, output):
    parent, official = parent.resolve(strict=True), official.resolve(strict=True)
    output = output.absolute()
    stage = output.with_name(output.name + '.processing')
    if output.exists() or stage.exists():
        raise FileExistsError('Choose a fresh dataset directory')
    original = json.loads((parent / 'manifest.json').read_text())
    addition = json.loads((official / 'manifest.json').read_text())
    if not original.get('complete') or not addition.get('complete') or len(addition['clips']) != 1:
        raise ValueError('Expected complete parent and one validated official Q1 clip')
    row = addition['clips'][0]
    proof = row['official_motion']
    qa_path = official / proof['validation']
    qa = json.loads(qa_path.read_text())
    if (row['robot'] != 'q1' or row['split'] != 'train' or row['frames'] != 500
            or row['five_point_source'] != 'robot_reference' or row['human_available']
            or sha(official / proof['path']) != SOURCE_SHA
            or proof['sha256'] != SOURCE_SHA or sha(qa_path) != proof['validation_sha256']
            or not qa['complete'] or not qa['original_joint_positions_preserved']
            or qa['source_sha256'] != SOURCE_SHA
            or sha(row['output']) != row['output_sha256'] or row['output_sha256'] != qa['output_sha256']
            or sha(row['five_point_sidecar']) != row['five_point_sha256']
            or row['five_point_sha256'] != qa['five_point_sha256']):
        raise ValueError('Official Q1 provenance or payload hash differs')
    if any((r['robot'], r['name']) == (row['robot'], row['name']) for r in original['clips']):
        raise ValueError('Official Q1 clip already exists')

    def relocate(value):
        if isinstance(value, dict):
            return {k: relocate(v) for k, v in value.items()}
        if isinstance(value, list):
            return [relocate(v) for v in value]
        if isinstance(value, str) and (value == str(parent) or value.startswith(str(parent) + '/')):
            return str(output) + value[len(str(parent)):]
        return value

    try:
        shutil.copytree(parent, stage, copy_function=link_or_copy)
        manifest = relocate(original)
        reference = Path('q1') / (row['name'] + '.npz')
        points = Path('robot_reference50') / reference
        points_target = stage / points
        points_target.parent.mkdir(parents=True, exist_ok=True)
        link_or_copy(row['output'], stage / reference)
        link_or_copy(row['five_point_sidecar'], points_target)
        link_or_copy(points_target, stage / 'keypoints50' / reference)
        provenance_dir = Path('official_sources/q1')
        shutil.copytree(official / 'raw', stage / provenance_dir / 'raw', copy_function=link_or_copy)
        link_or_copy(qa_path, stage / provenance_dir / qa_path.name)
        added = dict(row, output=str(output / reference),
                     five_point_sidecar=str(output / points), fps=50, alignment_eligible=False)
        added['official_motion'] = dict(proof, path=str(provenance_dir / proof['path']),
                                       validation=str(provenance_dir / proof['validation']))
        manifest['clips'].append(added)
        for robot in ('q1', 'bumi', 'g1'):
            for split in ('train', 'val'):
                subset = relocate(json.loads((parent / robot / (split + '.yaml')).read_text()))
                if robot == 'q1' and split == 'train':
                    subset['clips'].append(str(output / reference))
                expected = {r['output'] for r in manifest['clips'] if r['robot'] == robot and r['split'] == split}
                if set(subset['clips']) != expected:
                    raise ValueError('Subset membership differs from the prepared manifest')
                save_json(stage / robot / (split + '.yaml'), subset)
        review = relocate(json.loads((parent / 'standing-review.json').read_text()))
        save_json(stage / 'standing-review.json', review)
        manifest['neutral_review']['sha256'] = sha(stage / 'standing-review.json')
        manifest['official_recovery_parent'] = dict(path=str(parent / 'manifest.json'), sha256=sha(parent / 'manifest.json'))
        manifest['derivation']['official_recovery_append'] = dict(
            added_clips=['q1/' + row['name']], previous_clips_and_splits_preserved=True,
            native_bumi_g1_official_references_added=False)
        save_json(stage / 'manifest.json', manifest)
        digest = sha(stage / 'manifest.json')
        windows = relocate(json.loads((parent / 'windows.json').read_text()))
        window = dict(robot='q1', clip=row['name'], category='lying', source_id=row['source_id'],
                      start_frame=0, stop_frame=25, stand_frame=350, hold_end_frame=500)
        windows['windows'].append(window)
        windows['evaluation_cases'].append(dict(window, stop_frame=1, case_id='q1/lying/' + row['name']))
        windows.update(training_manifest_sha256=digest, reference_manifest_sha256=digest,
                       subset_sha256={r: sha(stage / r / 'train.yaml') for r in ('q1', 'bumi', 'g1')})
        windows['approved_target']['review_manifest']['sha256'] = manifest['neutral_review']['sha256']
        # shortcut: keep old reviewed neutral preview poses until official targets have their own review.
        save_json(stage / 'windows.json', windows)
        floor = relocate(json.loads((parent / 'contact-floor.json').read_text()))
        floor['prepared_manifest_sha256'] = digest
        floor['clips'].append(dict(robot='q1', name=row['name'], dataset=row['dataset'], split='train',
            path=added['output'], sha256=added['output_sha256'], valid=False, offset_m=0.,
            method='collision-clearance-only', reason='Per-frame collision lift is not a measured stationary sole calibration.'))
        floor['summary']['q1']['unknown'] += 1
        save_json(stage / 'contact-floor.json', floor)
        save_json(stage / 'official-recovery-append.json', dict(complete=True, manifest_sha256=digest,
            added=['q1/' + row['name']], original_clips_preserved=len(original['clips']),
            counts=dict(Counter(r['robot'] + '/' + r['split'] for r in manifest['clips'])),
            recovery_reset_share=.2, source_five_points='robot_reference', dynamics_qualified=False))
        os.rename(stage, output)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--official', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    append(args.parent, args.official, args.output)
