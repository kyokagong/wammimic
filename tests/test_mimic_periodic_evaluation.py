import json
from pathlib import Path

import pytest

from scripts.watch_mimic_evaluations import pin_checkpoint, result_rows


def test_pending_snapshot_survives_original_checkpoint_retention(tmp_path):
    for robot in ('q1','bumi','g1'):
        folder=tmp_path/robot
        folder.mkdir()
        (folder/'ckpt_15000.pt').write_bytes(robot.encode())
    snapshot=pin_checkpoint(tmp_path,15000,tmp_path/'pending')
    assert snapshot is not None
    receipt=json.loads((snapshot/'checkpoint-receipt.json').read_text())
    assert receipt['iteration']==15000
    for robot in ('q1','bumi','g1'):
        original=tmp_path/robot/'ckpt_15000.pt'
        pinned=snapshot/robot/original.name
        assert original.stat().st_ino==pinned.stat().st_ino
        original.unlink()
        assert pinned.read_bytes()==robot.encode()
    assert pin_checkpoint(tmp_path,15000,tmp_path/'pending')==snapshot


def test_snapshot_waits_for_every_robot(tmp_path):
    (tmp_path/'q1').mkdir()
    (tmp_path/'q1/ckpt_15000.pt').write_bytes(b'q1')
    assert pin_checkpoint(tmp_path,15000,tmp_path/'pending') is None
    assert not (tmp_path/'pending').exists()


def test_queued_evaluation_survives_removed_training_container(tmp_path,monkeypatch):
    from scripts import watch_mimic_evaluations as watcher
    from types import SimpleNamespace
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    parent=tmp_path/'parent.json';parent.write_text(json.dumps(dict(parent_iteration=12000)))
    (tmp_path/'config.json').write_text(json.dumps(dict(finetune_parent_receipt=str(parent),
        prepared_manifest=str(manifest),ckpt_every=5000,iterations=15000)))
    (tmp_path/'freeze.json').write_text(json.dumps(dict(runtime_image_id='test-image')))
    for robot in ('q1','bumi','g1'):
        (tmp_path/robot).mkdir()
        (tmp_path/robot/'ckpt_15000.pt').write_bytes(robot.encode())
    def unexpected_inspection(*args,**kwargs):
        pytest.fail('An active queued evaluation must not depend on its former training container')
    def stop_after_first_cycle(*args):
        raise SystemExit
    monkeypatch.setattr('sys.argv',['watch','--run',str(tmp_path)])
    monkeypatch.setattr(watcher.subprocess,'Popen',lambda *a,**kw:SimpleNamespace(poll=lambda:None))
    monkeypatch.setattr(watcher.subprocess,'run',unexpected_inspection)
    monkeypatch.setattr(watcher.subprocess,'check_output',unexpected_inspection)
    monkeypatch.setattr(watcher.time,'sleep',stop_after_first_cycle)
    with pytest.raises(SystemExit):
        watcher.main()
    state=json.loads((tmp_path/'periodic-evaluation/status.json').read_text())
    assert state['state']=='running' and state['evaluating']==15000


def test_curve_uses_every_held_out_clip_and_exact_checkpoint_identity(tmp_path):
    receipt=dict(checkpoints={r:dict(sha256=r) for r in ('q1','bumi','g1')})
    for robot,failure in [('q1',0.),('bumi',.3),('g1',.6)]:
        for interface in ('motion','human'):
            report=dict(iteration=15000,training_manifest_sha256='manifest',checkpoint_sha256=robot,
                weights='ema',num_clips=1139,physics=dict(zero_overflow_observed=True),failure_rate=failure,mean_norm_return=.5)
            (tmp_path/f'{robot}-{interface}-val.json').write_text(json.dumps(report))
    rows=result_rows(tmp_path,15000,receipt,'manifest')
    overall=next(r for r in rows if r['robot']=='all' and r['interface']=='motion')
    assert overall['clips']==3417
    assert overall['survival']==pytest.approx(.7)
    assert overall['mean_norm_return']==.5
    with pytest.raises(ValueError,match='identity'):
        result_rows(tmp_path,20000,receipt,'manifest')
    with pytest.raises(ValueError,match='identity'):
        result_rows(tmp_path,15000,receipt,'different manifest')
    report=json.loads((tmp_path/'q1-motion-val.json').read_text())
    report['num_clips']=2
    (tmp_path/'q1-motion-val.json').write_text(json.dumps(report))
    with pytest.raises(ValueError,match='membership'):
        result_rows(tmp_path,15000,receipt,'manifest')
    report.update(num_clips=1139,weights='raw')
    (tmp_path/'q1-motion-val.json').write_text(json.dumps(report))
    with pytest.raises(ValueError,match='identity'):
        result_rows(tmp_path,15000,receipt,'manifest')


def test_completed_robot_metrics_stream_before_whole_checkpoint_finishes(tmp_path):
    from scripts.publish_mimic import publish_event_files
    source=tmp_path/'evaluation/checkpoints/15000/metrics/q1/events.out.tfevents.test'
    source.parent.mkdir(parents=True)
    source.write_bytes(b'first')
    publish_event_files(tmp_path,tmp_path/'public')
    target=tmp_path/'public/evaluation/q1'/source.name
    assert target.stat().st_ino==source.stat().st_ino
    with source.open('ab') as stream:
        stream.write(b'second')
    assert target.read_bytes()==b'firstsecond'


def test_two_metric_table_separates_interfaces_and_marks_pending(tmp_path):
    from scripts.publish_mimic import publish_metric_tables
    folder=tmp_path/'evaluation/checkpoints/15000'
    folder.mkdir(parents=True)
    report=dict(iteration=15000,split='val',weights='ema',physics=dict(zero_overflow_observed=True),
                failure_rate=.25,mean_norm_return=.123456,num_clips=1139,
                checkpoint_sha256='q1',kp_horizon=10,five_point_source='mixed')
    (folder/'q1-motion-val.json').write_text(json.dumps(report))
    publish_metric_tables(tmp_path,tmp_path/'public')
    data=json.loads((folder/'paper-metrics.json').read_text())
    assert not data['complete'] and len(data['results'])==6
    assert data['weights']=='ema' and data['results'][0]['weights']=='ema'
    assert [(r['interface'],r['robot']) for r in data['results']]==[
        (i,r) for i in ('motion','human') for r in ('q1','bumi','g1')]
    assert data['results'][0]['survival']==.75
    assert 'survival' not in data['results'][3]
    text=(folder/'paper-metrics.md').read_text()
    assert '| Motion | Q1 | 75.000% | 0.1235 |' in text
    assert '| Human | Q1 | 评估中 | 评估中 |' in text
    assert '不是论文全动作共同训练库评估' in text
    events=list((tmp_path/'public/evaluation/table').glob('events.out.tfevents.*'))
    publish_metric_tables(tmp_path,tmp_path/'public')
    assert events==list((tmp_path/'public/evaluation/table').glob('events.out.tfevents.*'))
    report['iteration']=20000
    (folder/'q1-motion-val.json').write_text(json.dumps(report))
    with pytest.raises(ValueError,match='Invalid checkpoint'):
        publish_metric_tables(tmp_path,tmp_path/'public')
    report.update(iteration=15000,weights='raw')
    (folder/'q1-motion-val.json').write_text(json.dumps(report))
    with pytest.raises(ValueError,match='Invalid checkpoint'):
        publish_metric_tables(tmp_path,tmp_path/'public')
