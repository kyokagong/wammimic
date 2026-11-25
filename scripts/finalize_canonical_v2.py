"""Fit global Canonical Human v2.0 stats and run all 6 mandatory QA gates across all branches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from wammimic.data_pipeline.canonical_human import (
    CANONICAL_SCHEMA,
    fit_global_canonical_stats,
    run_six_gate_qa,
)
from wammimic.data_pipeline.common import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--qa-previews-dir", type=Path, default=None)
    parser.add_argument("--skip-hiw500", action="store_true")
    args = parser.parse_args()

    root = args.processed_root
    previews = args.qa_previews_dir or (root.parent / "qa_previews")
    previews.mkdir(parents=True, exist_ok=True)

    # Load human-protocol train splits (human-base + human-new)
    proto_path = root / "human-protocol/protocol.json"
    proto = json.loads(proto_path.read_text())
    hb_train = [r["episode_index"] for r in proto["episodes"] if r["source_id"] == 0 and r["split"] == "train"]
    hn_train = [r["episode_index"] for r in proto["episodes"] if r["source_id"] == 1 and r["split"] == "train"]
    n_hp_train = len(hb_train) + len(hn_train)

    gripper_splits = json.loads((root / "gripper/wam_lerobot_v2/g1_parallel_gripper/meta/splits.json").read_text())
    nymeria_splits = json.loads((root / "nymeria-40/wam_lerobot_v2/nymeria_ego/meta/splits.json").read_text())

    branch_specs = [
        {
            "name": "human_base",
            "dataset_dir": root / "human-base/wam_lerobot_v2/human_ego",
            "train_episodes": hb_train,
            "state_col": "observation.state",
            "weight": 0.25 * (len(hb_train) / max(n_hp_train, 1)),
            "has_ee": False,
        },
        {
            "name": "human_new",
            "dataset_dir": root / "human-new/wam_lerobot_v2/human_ego",
            "train_episodes": hn_train,
            "state_col": "observation.state",
            "weight": 0.25 * (len(hn_train) / max(n_hp_train, 1)),
            "has_ee": False,
        },
        {
            "name": "gripper",
            "dataset_dir": root / "gripper/wam_lerobot_v2/g1_parallel_gripper",
            "train_episodes": gripper_splits["train"],
            "state_col": "observation.state_5pt_hist",
            "weight": 0.25,
            "has_ee": True,
        },
        {
            "name": "nymeria_40",
            "dataset_dir": root / "nymeria-40/wam_lerobot_v2/nymeria_ego",
            "train_episodes": nymeria_splits["train"],
            "state_col": "observation.state",
            "weight": 0.25,
            "has_ee": False,
        },
    ]

    if not args.skip_hiw500 and (root / "hiw500/wam_lerobot_v2/g1_hiw500/meta/splits.json").is_file():
        hiw_splits = json.loads((root / "hiw500/wam_lerobot_v2/g1_hiw500/meta/splits.json").read_text())
        branch_specs.append({
            "name": "hiw500",
            "dataset_dir": root / "hiw500/wam_lerobot_v2/g1_hiw500",
            "train_episodes": hiw_splits["train"],
            "state_col": "observation.state_5pt_hist",
            "weight": 0.25,
            "has_ee": True,
        })

    global_stats_path = root / "canonical_human_5pt_stats.json"
    print(f"Fitting global Canonical Human v2.0 stats -> {global_stats_path} ...", flush=True)
    gstats = fit_global_canonical_stats(branch_specs, global_stats_path)
    print("Global stats fitted! Branch summary:", json.dumps(gstats["branches"], indent=2), flush=True)

    # Sync human-protocol/protocol.json to point to v2.0 global stats
    proto["canonical_version"] = CANONICAL_SCHEMA
    proto["global_stats_file"] = "../canonical_human_5pt_stats.json"
    proto["statistics"] = {
        "action": gstats["action_human"],
        "observation.state": gstats["observation.state"],
    }
    write_json(proto_path, proto)

    # Run all 6 QA gates on every branch (including read-only holdout human-operations)
    qa_targets = [
        ("human-base", root / "human-base", "human_ego", "observation.state"),
        ("human-new", root / "human-new", "human_ego", "observation.state"),
        ("human-operations", root / "human-operations", "human_ego", "observation.state"),
        ("gripper", root / "gripper", "g1_parallel_gripper", "observation.state_5pt_hist"),
        ("nymeria-40", root / "nymeria-40", "nymeria_ego", "observation.state"),
    ]
    if not args.skip_hiw500 and (root / "hiw500/wam_lerobot_v2/g1_hiw500/meta/splits.json").is_file():
        qa_targets.append(("hiw500", root / "hiw500", "g1_hiw500", "observation.state_5pt_hist"))

    summary = {"schema": CANONICAL_SCHEMA, "global_stats": str(global_stats_path), "branches": {}, "all_ready": True}
    for b_name, b_root, ds_name, st_col in qa_targets:
        rep_path = b_root / f"qa_reports/{b_name}_canonical_ready.json"
        sheet_path = previews / f"{b_name}_rgb_contact.png"
        print(f"Running 6-gate QA on {b_name} ...", flush=True)
        rep = run_six_gate_qa(
            b_name,
            b_root,
            ds_name,
            state_col=st_col,
            global_stats_path=global_stats_path,
            output_report_path=rep_path,
            contact_sheet_path=sheet_path,
        )
        summary["branches"][b_name] = {
            "ready": rep["ready"],
            "episodes_total": rep["episodes_total"],
            "gates": {k: v["pass"] for k, v in rep["gates"].items()},
            "medians": rep["gates"]["gate1_axes_and_signs"]["medians"],
            "max_abs_z_median": rep["gates"]["gate6_global_stats_alignment"]["max_abs_z_median"],
            "seam_anomalies": rep["gates"]["gate3_visual_and_seam"]["seam_anomalies"],
            "contact_sheet": str(sheet_path),
            "report": str(rep_path),
        }
        if not rep["ready"]:
            summary["all_ready"] = False
        print(f"  [{b_name}] ready={rep['ready']} gates={summary['branches'][b_name]['gates']} medians={summary['branches'][b_name]['medians']} max_z={summary['branches'][b_name]['max_abs_z_median']}", flush=True)

    summary_path = root / "canonical_v2_qa_summary.json"
    write_json(summary_path, summary)
    print(f"QA summary written to {summary_path} | all_ready={summary['all_ready']}", flush=True)


if __name__ == "__main__":
    main()
