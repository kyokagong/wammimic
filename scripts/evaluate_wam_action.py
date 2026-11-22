"""Score the final Action Expert checkpoint and render synchronized held-out replays."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from wammimic.data_pipeline.common import sha256, write_json
from wammimic.training.data import PilotDataset
from wammimic.training.evaluation import evaluate, pose_errors
from wammimic.training.model import CosmosWAM, load_backbone


def evaluate_run(run, config):
    training = json.loads((run / "training_summary.json").read_text())
    checkpoint = run / training["checkpoint"]
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not training["completed"] or state["step"] != config["steps"] or state["step"] != training["optimizer_updates"]:
        raise ValueError("Expected the final configured optimizer step")
    device = torch.device("cuda:0")
    model = CosmosWAM(load_backbone(config["backbone"], config["backbone_manifest"], device),
                      train_backbone=False).to(device)
    expected = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if state["expert"].keys() != expected:
        raise ValueError("Checkpoint does not match this Action Expert")
    model.load_state_dict(state["expert"], strict=False)
    metrics = {}
    reports = []
    for split in ("val", "test"):
        dataset = PilotDataset(config["protocol"], split, per_episode=4)
        for seed in (0, 1, 2):
            report = evaluate(model, None, dataset, device, seed=seed,
                              output=run / f"evaluation_{split}_seed{seed}.json",
                              save_predictions=seed == 0, diagnose_overlap=seed == 0)
            summary = {name: value["overall"] for name, value in report["metrics"].items()}
            reports.append({"split": split, "seed": seed, "metrics": summary})
            print(f"{split} seed={seed} ADE={summary['wam']['ade_mm']:.1f} mm", flush=True)
            if seed == 0:
                metrics[split] = summary["wam"]
    write_json(run / "evaluation_summary.json", {
        "step": state["step"], "checkpoint": training["checkpoint"], "checkpoint_sha256": sha256(checkpoint),
        "protocol_sha256": config["protocol_sha256"], "reports": reports,
    })
    with SummaryWriter(run / "evaluation-tensorboard") as writer:
        for split, values in metrics.items():
            for name in ("ade_mm", "fde_mm", "rotation_mean_deg"):
                writer.add_scalar(f"{split}/{name}", values[name], state["step"])


def render_run(run, config):
    from wammimic.training.visualization import case_video

    summary = json.loads((run / "evaluation_summary.json").read_text())
    protocol = json.loads(Path(config["protocol"]).read_text())
    directory = run / "preview"
    directory.mkdir(exist_ok=True)
    cases = []
    for split in ("val", "test"):
        report = json.loads((run / f"evaluation_{split}_seed0.json").read_text())
        records = report["records"]
        with np.load(run / f"evaluation_{split}_seed0.npz", allow_pickle=False) as arrays:
            predictions, references = arrays["prediction"], arrays["reference"]
        for scope in sorted({record["scope"] for record in records}):
            indexes = [index for index, record in enumerate(records) if record["scope"] == scope]
            worst = max(indexes, key=lambda index: pose_errors(
                predictions[index], references[index], records[index]["human_leg_length"],
            )["position_mm"].mean())
            for selection, index in (("fixed", indexes[0]), ("worst", worst)):
                destination = directory / f"{split}_{scope}_{selection}.mp4"
                case = case_video(protocol, records[index], predictions[index], references[index], destination)
                cases.append(case | {"split": split, "selection": selection})
                print(f"Rendered {destination.name}", flush=True)
    reports = [
        {"split": report["split"], "metrics": report["metrics"]["wam"]}
        for report in summary["reports"] if report["seed"] == 0
    ]
    write_json(directory / "manifest.json", {"step": summary["step"], "metrics": reports, "cases": cases})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("evaluate", "render"))
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    config = json.loads((args.run / "run.json").read_text())
    for file in ("protocol", "backbone_manifest"):
        if sha256(config[file]) != config[file + "_sha256"]:
            raise ValueError(f"Evaluation input changed since training: {file}")
    if args.stage == "evaluate":
        evaluate_run(args.run, config)
    else:
        render_run(args.run, config)


if __name__ == "__main__":
    main()
