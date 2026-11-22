import json
from pathlib import Path

import pytest

from scripts import append_q1_official_recovery as merger
from scripts.prepare_mimic_robot_five import save_json, sha


@pytest.fixture
def datasets(tmp_path, monkeypatch):
    parent, official = tmp_path / 'parent', tmp_path / 'official'
    parent.mkdir()
    rows = []
    for robot in ('q1', 'bumi', 'g1'):
        folder = parent / robot
        folder.mkdir()
        clip = folder / 'old.npz'
        clip.write_bytes(b'existing reference')
        rows.append(dict(robot=robot, name='old', output=str(clip), split='train'))
        save_json(folder / 'train.yaml', dict(root='/', clips=[str(clip)]))
        save_json(folder / 'val.yaml', dict(root='/', clips=[]))
    save_json(parent / 'standing-review.json', dict(path=str(parent / 'q1/old.npz')))
    save_json(parent / 'manifest.json', dict(complete=True, clips=rows,
        neutral_review=dict(sha256=sha(parent / 'standing-review.json')), derivation={}))
    save_json(parent / 'windows.json', dict(windows=[], evaluation_cases=[],
        approved_target=dict(review_manifest=dict(path=str(parent / 'standing-review.json')))))
    save_json(parent / 'contact-floor.json', dict(clips=[], summary=dict(q1=dict(unknown=0))))
    for folder in ('q1', 'robot_reference50/q1', 'raw'):
        (official / folder).mkdir(parents=True)
    (parent / 'keypoints50/q1').mkdir(parents=True)
    raw = official / 'raw/q1_getUp.npz'
    raw.write_bytes(b'official source fixture')
    target = official / 'q1/official_q1_getup.npz'
    target.write_bytes(b'new reference fixture')
    point = official / 'robot_reference50/q1/official_q1_getup.npz'
    point.write_bytes(b'new points fixture')
    monkeypatch.setattr(merger, 'SOURCE_SHA', sha(raw))
    qa = official / 'official-validation.json'
    save_json(qa, dict(complete=True, original_joint_positions_preserved=True,
        source_sha256=sha(raw), output_sha256=sha(target), five_point_sha256=sha(point)))
    row = dict(robot='q1', name='official_q1_getup', frames=500, split='train', dataset='primebot_official',
        source_id='official_fixture', five_point_source='robot_reference', human_available=False,
        output=str(target), output_sha256=sha(target), five_point_sidecar=str(point), five_point_sha256=sha(point),
        official_motion=dict(path='raw/q1_getUp.npz', sha256=sha(raw), validation=qa.name, validation_sha256=sha(qa)))
    save_json(official / 'manifest.json', dict(complete=True, clips=[row]))
    return parent, official, tmp_path / 'output'


def test_append_preserves_existing_payload_and_validation_split(datasets):
    parent, official, output = datasets
    original_manifest = sha(parent / 'manifest.json')
    merger.append(parent, official, output)
    result = json.loads((output / 'manifest.json').read_text())
    assert len(result['clips']) == 4
    assert sha(parent / 'manifest.json') == original_manifest
    for robot in ('q1', 'bumi', 'g1'):
        assert (output / robot / 'old.npz').stat().st_ino == (parent / robot / 'old.npz').stat().st_ino
        assert json.loads((output / robot / 'val.yaml').read_text())['clips'] == []
    assert Path(result['clips'][-1]['output']).is_file()
    assert (output / 'keypoints50/q1/official_q1_getup.npz').read_bytes() == b'new points fixture'
    windows = json.loads((output / 'windows.json').read_text())
    assert windows['training_manifest_sha256'] == sha(output / 'manifest.json')
    assert windows['windows'][-1]['start_frame'] == 0
    assert windows['windows'][-1]['hold_end_frame'] == 500
    floor = json.loads((output / 'contact-floor.json').read_text())
    assert floor['clips'][-1]['valid'] is False


def test_corrupt_points_are_rejected_before_copying(datasets):
    parent, official, output = datasets
    (official / 'robot_reference50/q1/official_q1_getup.npz').write_bytes(b'changed')
    with pytest.raises(ValueError, match='payload hash differs'):
        merger.append(parent, official, output)
    assert not output.exists()
    assert not output.with_name(output.name + '.processing').exists()
