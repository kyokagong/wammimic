"""Freeze the Global + Ref. Mimic base run or its 2,000-step flip continuation."""

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from morphmimic.train.recovery import file_sha
from morphmimic.train.run_identity import ManifestIdentity, create_receipt, read_receipt, write_receipt
from scripts import train_optimized


ITERATIONS = 10_000
CHECKPOINT_EVERY = 1_000
KEEP_CHECKPOINTS = 4


def flip_continuation(run, parent, identity, manifest, freeze):
    import numpy as np

    parent = parent.resolve(strict=True)
    config = json.loads((parent / "config.json").read_text())
    if config["iterations"] != ITERATIONS or config["prepared_manifest"] != str(identity.path):
        raise ValueError("Flip continuation requires the 10,000-step model on the same data")
    windows = []
    for row in manifest["clips"]:
        if row["split"] != "train" or row["dataset"] != "seed_flips":
            continue
        if file_sha(row["output"]) != row["output_sha256"] or row["fps"] != 50:
            raise ValueError(f"Flip reference identity differs for {row['name']}")
        with np.load(row["output"], allow_pickle=False) as reference:
            root = reference["body_names"].tolist().index(str(reference["root_body"].item()))
            height = reference["body_pos_w"][:, root, 2]
        phase = row["neutral_contract"]
        start, stop = phase["source_start_frame"], phase["source_stop_frame"]
        if len(height) != row["frames"] or not np.isfinite(height).all() or stop - start < 40:
            raise ValueError(f"Invalid flip reference {row['name']}")
        peak = start + int(height[start:stop].argmax())
        candidates = np.arange(max(start + 2, peak - 40), peak - 1)
        ascent = height[candidates + 2] - height[candidates - 2]
        if not len(ascent) or ascent.max() <= 0:
            raise ValueError(f"Missing rising phase in flip {row['name']}")
        rise = int(candidates[ascent.argmax()])
        windows.append(dict(robot=row["robot"], clip=row["name"],
                            start_frame=max(start, rise - 25), stop_frame=min(stop, peak + 15),
                            ascent_frame=rise, peak_frame=peak))
    robots = config["robots"]
    assert all(sum(row["robot"] == robot for row in windows) == 4 for robot in robots)
    focus = run / "flip-windows.json"
    focus.write_text(json.dumps(dict(schema_version=1, fps=50, training_manifest_sha256=identity.sha256,
        subset_sha256={robot: file_sha(path) for robot, path in zip(robots, config["subset_yamls"], strict=True)},
        windows=windows), indent=2))
    receipt = run / "parent-10000.json"
    checkpoints = {}
    for robot in robots:
        checkpoint = parent / robot / "ckpt_10000.pt"
        checkpoints[robot] = dict(path=str(checkpoint), sha256=file_sha(checkpoint),
                                  freeze_receipt=str(parent / "freeze.json"))
    receipt.write_text(json.dumps(dict(schema_version=1, parent_iteration=10_000, target_iteration=12_000,
        prepared_manifest_sha256=identity.sha256, windows_sha256=config["recovery_windows_sha256"],
        checkpoints=checkpoints), indent=2))
    config.update(iterations=12_000, run_name=run.name, log_dir=str(run.parent), freeze_receipt=str(freeze),
                  importance_sampling=True, recovery_reset_share=.20, probe_focus_dir=None,
                  finetune_parent_receipt=str(receipt), finetune_parent_receipt_sha256=file_sha(receipt),
                  flip_windows=str(focus), flip_windows_sha256=file_sha(focus), flip_reset_share=.10)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "train"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--gpu-ids", type=int, nargs=3, default=[0, 2, 3])
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--parent-run", type=Path, help="Prepare the 10,000 to 12,000 step flip continuation")
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    parser.add_argument("--ckpt-every", type=int, default=CHECKPOINT_EVERY)
    parser.add_argument("--keep-checkpoints", type=int, default=KEEP_CHECKPOINTS)
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--importance-sampling", action="store_true")
    parser.add_argument("--uniform-clip-base", action="store_true")
    parser.add_argument("--no-focused-resets", action="store_true")
    args = parser.parse_args()

    from morphmimic.train.trainer import TrainCfg

    run = args.run.resolve(strict=True)
    data = args.data_root.resolve(strict=True)
    image = os.environ["BLINKMORPH_RUNTIME_IMAGE_ID"]
    source = Path(train_optimized.__file__).resolve().parents[1]
    config_path = run / "config.json"
    freeze = run / "freeze.json"
    identity = ManifestIdentity(data / "manifest.json")
    if args.mode == "prepare":
        if min(args.iterations, args.ckpt_every, args.keep_checkpoints, args.num_envs) <= 0:
            raise ValueError("Training steps, checkpoint cadence, retention and environment count must be positive")
        if args.uniform_clip_base and not args.importance_sampling:
            raise ValueError("Uniform importance base requires importance sampling")
        manifest = json.loads(identity.path.read_text())
        if manifest.get("complete") is not True:
            raise ValueError("Prepared training manifest is incomplete")
        windows = data / "windows.json"
        point_root = manifest.get("five_point_root", manifest["human_root"])
        values = train_optimized.config_values(data, point_root, run.parent, run.name, args.gpu_ids)
        if "five_point_root" in manifest:
            values.update(kp_fixed_horizon=10, kp_preview_mix=False)
        values.update(
            iterations=args.iterations, lr_decay_iters=args.iterations, num_envs=args.num_envs,
            ckpt_every=args.ckpt_every, keep_checkpoints=args.keep_checkpoints,
            freeze_receipt=str(freeze), recovery_windows=str(windows), recovery_windows_sha256=file_sha(windows),
            recovery_reset_share=0.2, importance_sampling=args.importance_sampling, ref_feedback=True,
        )
        if args.uniform_clip_base:
            values['importance_source_balance'] = False
        if args.no_focused_resets:
            values.update(recovery_reset_share=0., flip_reset_share=0., probe_reset_share=0., probe_focus_dir=None)
        config = asdict(TrainCfg(**values))
        if args.parent_run:
            config = asdict(TrainCfg(**flip_continuation(run, args.parent_run, identity, manifest, freeze)))
        receipt = create_receipt(source, image, config, identity.sha256)
        with config_path.open("x") as stream:
            json.dump(config, stream, indent=2)
        write_receipt(freeze, receipt)
        return

    config = json.loads(config_path.read_text())
    read_receipt(freeze, identity.sha256, source_root=source, image_id=image, config=config)
    if not args.resume and any((run / robot).exists() for robot in config["robots"]):
        raise FileExistsError("Run already has training state. Use a fixed checkpoint to resume.")
    from morphmimic.scripts.gpu_guard import select_gpus
    from morphmimic.scripts.train import launch

    select_gpus(config["gpu_ids"], True)
    launch(TrainCfg(**config), str(args.resume.resolve(strict=True)) if args.resume else None)


if __name__ == "__main__":
    main()
