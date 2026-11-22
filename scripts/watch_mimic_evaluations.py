"""Pin saved checkpoint sets and evaluate them serially on the spare GPU."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROBOTS = ('q1', 'bumi', 'g1')
INTERFACES = ('motion', 'human')


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def pin_checkpoint(run, step, pending):
    target = pending / str(step)
    if (target / 'checkpoint-receipt.json').is_file():
        return target
    sources = {robot: run / robot / f'ckpt_{step}.pt' for robot in ROBOTS}
    if any(p.is_symlink() for p in sources.values()):
        raise ValueError('Checkpoint pins require regular files')
    if not all(p.is_file() for p in sources.values()):
        return None
    stage = target.with_name(target.name + '.pinning')
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    checks = {}
    try:
        for robot, source in sources.items():
            folder = stage / robot
            folder.mkdir()
            pinned = folder / source.name
            os.link(source, pinned)
            checks[robot] = dict(source=str(source), sha256=sha(pinned))
        save_json(stage / 'checkpoint-receipt.json', dict(iteration=step, checkpoints=checks))
        os.rename(stage, target)
    except FileNotFoundError:
        shutil.rmtree(stage)
        return None
    return target


def result_rows(output, step, receipt, manifest_sha):
    rows = []
    for robot in ROBOTS:
        for interface in INTERFACES:
            report = json.loads((output / f'{robot}-{interface}-val.json').read_text())
            if (report['iteration'] != step or report['training_manifest_sha256'] != manifest_sha
                    or report.get('weights') != 'ema'
                    or report['checkpoint_sha256'] != receipt['checkpoints'][robot]['sha256']
                    or report['num_clips'] != 1139 or not report['physics']['zero_overflow_observed']):
                raise ValueError('Evaluation curve identity, full validation membership or physics differs')
            rows.append(dict(step=step, robot=robot, interface=interface, clips=report['num_clips'],
                survival=1-report['failure_rate'], mean_norm_return=report['mean_norm_return'],
                checkpoint_sha256=report['checkpoint_sha256']))
    for interface in INTERFACES:
        selected = [row for row in rows if row['interface'] == interface]
        count = sum(row['clips'] for row in selected)
        rows.append(dict(step=step, robot='all', interface=interface, clips=count,
            survival=sum(row['survival']*row['clips'] for row in selected)/count,
            mean_norm_return=sum(row['mean_norm_return']*row['clips'] for row in selected)/count,
            checkpoint_sha256=''))
    return rows


def publish_result(run, output, rows):
    from torch.utils.tensorboard import SummaryWriter
    metrics = run / 'evaluation/metrics'
    for robot in ROBOTS:
        destination = metrics / robot
        destination.mkdir(parents=True, exist_ok=True)
        for source in (output / 'metrics' / robot).glob('events.out.tfevents.*'):
            target = destination / source.name
            if not target.exists():
                os.link(source, target)
    with SummaryWriter(str(metrics / 'aggregate')) as writer:
        for row in rows:
            if row['robot'] == 'all':
                for metric in ('survival', 'mean_norm_return'):
                    writer.add_scalar(f"val/{row['interface']}/{metric}", row[metric], row['step'])
    save_json(output / 'curve-point.json', dict(complete=True, step=rows[0]['step'], rows=rows))
    points = []
    for point in sorted((run / 'evaluation/checkpoints').glob('*/curve-point.json')):
        points.extend(json.loads(point.read_text())['rows'])
    points.sort(key=lambda row: (row['step'], row['robot'], row['interface']))
    save_json(run / 'evaluation/checkpoint-curve.json', dict(rows=points, seed=0,
        protocol='All 1139 validation clips per robot, Motion and mixed five-point, same native benchmark.'))
    temporary = run / 'evaluation/checkpoint-curve.csv.tmp'
    with temporary.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(points[0]))
        writer.writeheader(); writer.writerows(points)
    os.replace(temporary, run / 'evaluation/checkpoint-curve.csv')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--train-container', default='wammimic-mimic-all-motion-40000')
    args = parser.parse_args()
    run = args.run.resolve(strict=True)
    config = json.loads((run / 'config.json').read_text())
    image = json.loads((run / 'freeze.json').read_text())['runtime_image_id']
    parent = json.loads(Path(config['finetune_parent_receipt']).read_text())['parent_iteration']
    steps = list(range((parent//config['ckpt_every']+1)*config['ckpt_every'], config['iterations']+1, config['ckpt_every']))
    if config['iterations'] not in steps:
        steps.append(config['iterations'])
    manifest_sha = sha(Path(config['prepared_manifest']))
    root = run / 'periodic-evaluation'
    pending = root / 'pending'
    completed, active = set(), None
    for step in steps:
        point = run / f'evaluation/checkpoints/{step}/curve-point.json'
        if point.is_file():
            completed.add(step)
            if (pending / str(step)).exists():
                shutil.rmtree(pending / str(step))
    try:
        while True:
            pins = {}
            for step in steps:
                if step not in completed:
                    target = pin_checkpoint(run, step, pending)
                    if target:
                        pins[step] = target
            if active and active['process'].poll() is not None:
                active['log'].close()
                if active['process'].returncode:
                    raise RuntimeError(f"Checkpoint {active['step']} evaluation failed; see {active['output'] / 'worker.log'}")
                step, output = active['step'], active['output']
                receipt = json.loads((pending / str(step) / 'checkpoint-receipt.json').read_text())
                rows = result_rows(output, step, receipt, manifest_sha)
                shutil.copy2(pending / str(step) / 'checkpoint-receipt.json', output / 'checkpoint-receipt.json')
                publish_result(run, output, rows)
                subprocess.run(['docker','run','--rm','--network','none','-v',f'{run}:{run}:rw',
                    '--entrypoint','python',image,str(run/'tools/plot_mimic_evaluation_curve.py'),
                    '--run',str(run)],check=True)
                if step == config['iterations']:
                    for source in output.iterdir():
                        if source.is_file() and source.suffix in ('.json','.csv','.yaml','.md'):
                            shutil.copy2(source, run / 'evaluation' / source.name)
                shutil.rmtree(pending / str(step))
                completed.add(step)
                active = None
            if active is None:
                ready = sorted(set(pins)-completed)
                if ready:
                    step = ready[0]
                    output = run / f'evaluation/checkpoints/{step}'
                    output.mkdir(parents=True, exist_ok=True)
                    log = (output / 'worker.log').open('a')
                    process = subprocess.Popen(['bash',str(run/'tools/evaluate_checkpoint.sh'),str(step),str(pending/str(step)),str(output)],stdout=log,stderr=subprocess.STDOUT)
                    active = dict(step=step,output=output,process=process,log=log)
            status = dict(state='complete' if len(completed)==len(steps) else 'running', gpu=1,
                expected_steps=steps, completed_steps=sorted(completed),
                evaluating=active['step'] if active else None, pending_steps=sorted(set(pins)-completed))
            save_json(root / 'status.json', status)
            if status['state']=='complete':
                return
            if active is None and not (set(pins)-completed):
                running = subprocess.run(['docker','inspect','-f','{{.State.Running}}',args.train_container],
                    text=True, capture_output=True)
                if running.returncode or running.stdout.strip() != 'true':
                    raise RuntimeError('Training stopped before every expected checkpoint was available')
            time.sleep(5)
    except Exception as error:
        save_json(root/'status.json',dict(state='failed',error=str(error),completed_steps=sorted(completed),evaluating=active['step'] if active else None))
        raise


if __name__ == '__main__':
    main()
