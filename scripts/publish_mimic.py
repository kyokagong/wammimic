"""Project this run's TensorBoard event files into a read-only web service mount."""

import argparse
import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
import time

from tensorboard.compat.proto.event_pb2 import Event, SessionLog
from tensorboard.summary.writer.record_writer import RecordWriter


def mark_initial_session(destination):
    marker = destination / "events.out.tfevents.0000000000.initial"
    if marker.exists():
        return
    # TensorBoard treats its first START as initialization and only purges on subsequent STARTs.
    with NamedTemporaryFile(dir=destination) as stream:
        writer = RecordWriter(stream)
        writer.write(Event(file_version="brain.Event:2").SerializeToString())
        writer.write(Event(session_log=SessionLog(status=SessionLog.START)).SerializeToString())
        writer.flush()
        os.link(stream.name, marker)


def publish_event_files(run, metrics_root=None):
    metrics_root = metrics_root or (run / "public-metrics")
    for robot in ("q1", "bumi", "g1", "aggregate"):
        for group, source in (("training", run / robot),
                              ("evaluation", run / "evaluation" / "metrics" / robot),
                              ("baseline", run / "evaluation" / "baseline" / "metrics" / robot)):
            if robot == "aggregate" and group != "evaluation":
                continue
            if group == "baseline" and not source.is_dir():
                continue
            destination = metrics_root / group / robot
            destination.mkdir(parents=True, exist_ok=True)
            if group == "training":
                mark_initial_session(destination)
            events = list(source.glob("events.out.tfevents.*"))
            if group == "evaluation" and robot != "aggregate":
                events.extend((run / "evaluation/checkpoints").glob(f"*/metrics/{robot}/events.out.tfevents.*"))
            for event in events:
                if event.is_symlink() or not event.is_file():
                    continue
                try:
                    os.link(event, destination / event.name, follow_symlinks=False)
                except FileExistsError:
                    pass


def publish_metric_tables(run, metrics_root):
    """Keep incomplete checkpoint tables explicit; never fill missing runs with zero."""
    for output in sorted((run / 'evaluation/checkpoints').glob('[0-9]*')):
        step = int(output.name)
        rows = []
        for interface in ('motion', 'human'):
            for robot in ('q1', 'bumi', 'g1'):
                path = output / f'{robot}-{interface}-val.json'
                row = dict(interface=interface, robot=robot, status='evaluating')
                if path.is_file():
                    report = json.loads(path.read_text())
                    survival, value = 1 - report['failure_rate'], report['mean_norm_return']
                    if (report['iteration'] != step or report['split'] != 'val'
                            or report['weights'] != 'ema'
                            or not report['physics']['zero_overflow_observed']
                            or not 0 <= survival <= 1 or not math.isfinite(value)):
                        raise ValueError(f'Invalid checkpoint metric table input: {path}')
                    row.update(status='complete', clips=report['num_clips'], survival=survival,
                               mean_norm_return=value, checkpoint_sha256=report['checkpoint_sha256'],
                               weights=report['weights'],
                               five_point_source=report.get('five_point_source'),
                               kp_horizon=report.get('kp_horizon'))
                rows.append(row)
        result = dict(iteration=step, weights='ema', complete=all(r['status'] == 'complete' for r in rows), results=rows,
                      scope='validation', human_source='mixed original_human and robot_reference')
        target = output / 'paper-metrics.json'
        encoded = json.dumps(result, indent=2, allow_nan=False) + '\n'
        if target.is_file() and target.read_text() == encoded:
            continue
        lines = [f'# {step:,} 步 Motion / Human 评估 · EMA 权重', '',
                 '范围：各机型固定留出集；Human 为当前混合来源五点入口（原始人体 / 机器人参考）。',
                 '这是本地留出评估，不是论文全动作共同训练库评估。', '',
                 '| 输入接口 | 本体 | 整段完成率 Survival ↑ | 平均归一化回报 ↑ |',
                 '| --- | --- | ---: | ---: |']
        for row in rows:
            values = (f"{row['survival']:.3%}", f"{row['mean_norm_return']:.4f}") if row['status'] == 'complete' else ('评估中', '评估中')
            lines.append(f"| {row['interface'].title()} | {row['robot'].upper()} | {values[0]} | {values[1]} |")
        lines.extend(['', 'Survival = 到达最后一帧且未提前终止的动作数 / 全部动作数。',
                      '平均归一化回报 = 各动作 Σr / [0.1 × (完整动作帧数 − 1)] 的算术平均；提前终止后未执行部分不补奖励。',
                      'Motion 和 Human 分开统计；起身回归、跟踪误差和训练短窗口统计不进入上表。', '',
                      'Human 预览帧数：' + '；'.join(f"{r['robot']} H{r['kp_horizon']}" for r in rows if r['interface'] == 'human' and r['status'] == 'complete') + '。',
                      '已完成动作数：' + '；'.join(f"{r['interface']}/{r['robot']} {r['clips']}" for r in rows if r['status'] == 'complete') + '。'])
        markdown = '\n'.join(lines) + '\n'
        temporary = target.with_suffix('.tmp')
        temporary.write_text(encoded)
        os.replace(temporary, target)
        (output / 'paper-metrics.md').write_text(markdown)
        from torch.utils.tensorboard import SummaryWriter
        with SummaryWriter(str(metrics_root / 'evaluation/table')) as writer:
            writer.add_text('Motion_Human/Survival_and_return', markdown, step)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--metrics-root", type=Path)
    args = parser.parse_args()
    metrics_root = args.metrics_root or (args.run / 'public-metrics')
    while True:
        publish_event_files(args.run, metrics_root)
        publish_metric_tables(args.run, metrics_root)
        time.sleep(10)


if __name__ == "__main__":
    main()
