"""Run native recovery regression and held-out tracking for the completed Mimic run."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from morphmimic.train.recovery import file_sha
from morphmimic.train.run_identity import ManifestIdentity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--robot", choices=("q1", "bumi", "g1"))
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--iteration", type=int, help="Evaluate this saved training step")
    parser.add_argument("--checkpoint-root", type=Path, help="Pinned checkpoint set with q1/bumi/g1 subdirectories")
    parser.add_argument("--tracking-only", action="store_true")
    parser.add_argument("--five-point-source", choices=("mixed", "original_human", "robot_reference"), default="mixed")
    args = parser.parse_args()

    run = args.run.resolve(strict=True)
    config = json.loads((run / "config.json").read_text())
    freeze = json.loads((run / "freeze.json").read_text())
    manifest = ManifestIdentity(config["prepared_manifest"])
    if manifest.sha256 != freeze["data_manifest_sha256"]:
        raise ValueError("Evaluation manifest differs from the frozen training manifest")
    data = manifest.path.parent
    dataset = json.loads(manifest.path.read_text())
    rows = dataset["clips"]
    iteration = args.iteration if args.iteration is not None else config["iterations"]
    if not 0 < iteration <= config["iterations"]:
        raise ValueError("Evaluation step must be within the configured training range")
    checkpoint_root = args.checkpoint_root.resolve(strict=True) if args.checkpoint_root else run
    out = args.output or (run / "evaluation")
    out.mkdir(parents=True, exist_ok=True)

    from morphmimic.scripts.gpu_guard import select_gpu

    select_gpu(args.gpu)
    import pandas as pd
    from mjlab.utils.torch import configure_torch_backends
    from morphmimic.data.manifest import load_subset
    from morphmimic.eval.benchmark import GLOBAL_RESULT_COLUMNS, evaluate, summarize
    from morphmimic.scripts.eval import checkpoint_identity, verify_checkpoint
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from torch.utils.tensorboard import SummaryWriter

    configure_torch_backends()
    for robot in (args.robot,) if args.robot else config["robots"]:
        checkpoint = checkpoint_root / robot / f"ckpt_{iteration}.pt"
        identity = checkpoint_identity(checkpoint)
        import torch
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if (saved["iteration"] != iteration or saved["cfg"]["robot"] != robot
                or saved["optimized_training"]["prepared_manifest_sha256"] != manifest.sha256):
            raise ValueError("Checkpoint step, robot or training manifest differs")
        del saved
        group = {row["output"]: row for row in rows if row["robot"] == robot and row["split"] == "val"}
        val = load_subset(str(data / robot / "val.yaml"))
        if not val or len(val) != len(group) or set(val) != group.keys():
            raise ValueError(f"{robot}: validation subset differs from the manifest")
        metrics_dir = out / "metrics" / robot
        published = set()
        if metrics_dir.exists():
            events = EventAccumulator(str(metrics_dir), size_guidance={"scalars": 0}).Reload()
            published = {(tag, event.step) for tag in events.Tags()["scalars"] for event in events.Scalars(tag)}
        with SummaryWriter(str(metrics_dir)) as writer:
            def scalar(tag, value):
                if (tag, iteration) not in published:
                    writer.add_scalar(tag, value, iteration)
                    published.add((tag, iteration))

            for interface in ("motion", "human"):
                recovery = out / f"{robot}-{interface}-recovery.json"
                if not args.tracking_only and not recovery.is_file():
                    subprocess.run([
                        sys.executable, "-m", "scripts.evaluate_recovery", "--ckpt", str(checkpoint),
                        "--parent-iteration", str(iteration), "--robot", robot, "--interface", interface,
                        "--subset-yaml", str(data / robot / "train.yaml"), "--windows", config["recovery_windows"],
                        "--standing-target", str(data / "standing-review.json"), "--output", str(recovery),
                        "--gpu", str(args.gpu),
                    ], check=True)
                if not args.tracking_only:
                    report = json.loads(recovery.read_text())
                    if report["checkpoint"]["sha256"] != identity["checkpoint_sha256"] or report["criteria"]["hold_seconds"] != 3:
                        raise ValueError(f"Unexpected recovery report {recovery}")
                    for category, metrics in report["categories"].items():
                        for name in ("success_rate", "mean_time_to_stand_s"):
                            if metrics[name] is not None:
                                scalar(f"recovery/{interface}/{category}/{name}", metrics[name])
                    writer.flush()

                label = interface
                if interface == "human" and args.five_point_source != "mixed":
                    label += "-" + args.five_point_source
                result = out / f"{robot}-{label}-val.json"
                if not result.is_file():
                    subset = out / f"{robot}-{label}-val.yaml"
                    selected = [group[path] for path in val if interface == "motion" or (
                        group[path].get("five_point_available", group[path]["human_available"])
                        and (args.five_point_source == "mixed" or
                             group[path].get("five_point_source", "original_human") == args.five_point_source))]
                    if not selected:
                        raise ValueError("No validation clips for the selected five-point source")
                    for row in selected:
                        if file_sha(row["output"]) != row["output_sha256"]:
                            raise ValueError(f"Motion bytes changed for {row['output']}")
                        if interface == "human":
                            side = row.get("five_point_sidecar", row.get("human_sidecar"))
                            digest = row.get("five_point_sha256", row.get("human_sha256"))
                            if file_sha(side) != digest:
                                raise ValueError(f"Five-point bytes changed for {side}")
                    subset.write_text(json.dumps({"root": "/", "clips": [row["output"] for row in selected]}))
                    # The loader auto-discovers mixed sidecars unless motion evaluation supplies an empty root.
                    with TemporaryDirectory(prefix="motion-sidecars-") as no_sidecars:
                        frame = evaluate(checkpoint, str(subset), "cuda:0", num_envs=256, robot=robot,
                                         interface=interface, human_points="five", kp_horizon=config.get("kp_fixed_horizon", 1),
                                         library_dtype=config["library_dtype"],
                                         human_root=dataset.get("five_point_root", config["human_root"]) if interface == "human" else no_sidecars,
                                         goal_anchor=config["goal_anchor"], control_graph=True)
                    if len(frame) != len(selected) or set(frame["name"]) != {row["name"] for row in selected}:
                        raise ValueError(f"Unexpected evaluation clip membership in {subset}")
                    verify_checkpoint(identity)
                    report = summarize(frame, pd.DataFrame([
                        {"name": row["name"], "category": row["dataset"]} for row in selected
                    ]))
                    report.update(identity, iteration=iteration, training_manifest_sha256=manifest.sha256,
                                  physics=frame.attrs["physics"], split="val", five_point_source=args.five_point_source,
                                  kp_horizon=config.get("kp_fixed_horizon", 1))
                    if interface == "human":
                        sources = {row["name"]: row.get("five_point_source", "original_human") for row in selected}
                        frame["five_point_source"] = frame["name"].map(sources)
                        report["five_point_groups"] = {
                            source: dict(num_clips=len(part), survival=float((~part["failed"]).mean()),
                                         mean_norm_return=float(part["norm_return"].mean()))
                            for source, part in frame.groupby("five_point_source")
                        }
                    report["global_tracking"] = {
                        name: float(frame[name].mean()) if frame[name].notna().any() else None
                        for name in GLOBAL_RESULT_COLUMNS
                    }
                    frame.to_csv(result.with_suffix(".csv"), index=False)
                    temporary = result.with_suffix(".tmp")
                    temporary.write_text(json.dumps(report, indent=2))
                    temporary.replace(result)
                report = json.loads(result.read_text())
                if (report["checkpoint_sha256"] != identity["checkpoint_sha256"]
                        or report["iteration"] != iteration or report["training_manifest_sha256"] != manifest.sha256):
                    raise ValueError(f"Unexpected validation report {result}")
                scalar(f"val/{label}/survival", 1. - report["failure_rate"])
                for metric in ("failure_rate", "mean_norm_return"):
                    scalar(f"val/{label}/{metric}", report[metric])
                for metric in ("mean_xy", "final_xy", "mean_abs_yaw_deg"):
                    scalar(f"val/{label}/drift/{metric}", report["drift"][metric])
                for metric, value in report["global_tracking"].items():
                    if value is not None:
                        scalar(f"val/{label}/global/{metric}", value)
                writer.flush()
                print(f"[mimic-eval] {robot} {interface}: {report['num_clips']} val clips", flush=True)


if __name__ == "__main__":
    main()
