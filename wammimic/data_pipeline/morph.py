"""Import existing verified robot FK trajectories without re-running retargeting."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import os
import numpy as np

from .common import sha256, write_json
from .geometry import LEG_LENGTH, matrices, mimic_window


def reference_floors(manifest_path,document):
    path=Path(manifest_path).with_name("contact-floor.json")
    receipt=json.loads(path.read_text())
    if receipt.get("prepared_manifest_sha256")!=sha256(manifest_path):
        raise ValueError("Reference-floor manifest identity differs")
    sources={(r["robot"],r["name"]):r for r in document["clips"]};floors={}
    for row in receipt["clips"]:
        key=(row["robot"],row["name"]);source=sources.get(key)
        if key in floors or source is None or row["sha256"]!=source["output_sha256"] or row["path"]!=source["output"]:
            raise ValueError("Reference-floor source identity differs")
        if type(row["valid"]) is not bool or not isinstance(row["offset_m"],(int,float)) or not np.isfinite(row["offset_m"]) or abs(row["offset_m"])>.12 or (not row["valid"] and row["offset_m"]!=0):
            raise ValueError("Invalid reference-floor estimate")
        floors[key]=row
    if floors.keys()!=sources.keys():raise ValueError("Missing reference-floor calibration")
    return floors,sha256(path)


def import_morph(manifest_path, source_prefix, mounted_root, output_root, workers=4):
    manifest_path, output_root = Path(manifest_path), Path(output_root)
    document = json.loads(manifest_path.read_text())
    if document.get("complete") is not True:
        raise ValueError("Incomplete source manifest")
    identifiers=[(r["robot"],r["name"]) for r in document["clips"]]
    if len(identifiers)!=len(set(identifiers)):
        raise ValueError("Duplicate robot/trajectory identity")
    floors,floor_sha=reference_floors(manifest_path,document)
    output_root.mkdir(parents=True, exist_ok=True)
    identity = {"schema": "wammimic-morph-v1", "source_manifest_sha256": sha256(manifest_path)}
    marker = output_root / "source.json"
    if marker.exists() and json.loads(marker.read_text()) != identity:
        raise ValueError("Output belongs to a different source manifest")
    if not marker.exists() and any(output_root.iterdir()):
        raise FileExistsError("Refuse nonempty unowned output")
    write_json(marker, identity)

    def convert(row):
        robot, name = row["robot"], row["name"]
        if robot not in LEG_LENGTH or Path(name).name != name:
            raise ValueError("Unsafe trajectory identity")
        source = Path(mounted_root) / Path(row["output"]).relative_to(source_prefix)
        expected = row["output_sha256"]
        if sha256(source) != expected:
            raise ValueError(f"Source hash mismatch: {source}")
        relative = Path(robot) / name
        dest = output_root / relative
        receipt_path = dest / "receipt.json"
        if receipt_path.exists():
            saved = json.loads(receipt_path.read_text())
            if saved["input_sha256"] != expected:
                raise ValueError("Changed source on resume")
            for filename, digest in saved["output_sha256"].items():
                if sha256(dest / filename) != digest:
                    raise ValueError(f"Changed output: {dest / filename}")
            return saved
        with np.load(source, allow_pickle=False) as data:
            if int(data["fps"]) != 50 or len(data["joint_pos"]) < 10:
                raise ValueError("Expected a continuous 50 Hz clip with >=10 frames")
            names = data["body_names"].tolist()
            if len(names) != 14 or str(data["root_body"]) != names[0]:
                raise ValueError("Unsupported tracked-body convention")
            indices = [0, 3, 6, 10, 13]
            if not all("ankle" in names[i] for i in indices[1:3]):
                raise ValueError("Ankle ordering mismatch")
            poses = np.concatenate([data["body_pos_w"][:, indices], data["body_quat_w"][:, indices]], axis=-1)
            joints = np.stack([data["joint_pos"], data["joint_vel"]], axis=1)
            if not np.isfinite(joints).all() or not np.isfinite(poses).all():
                raise ValueError("Nonfinite trajectory")
            matrices(poses[..., 3:])
            for i in [0, len(poses)//2, len(poses)-1]:
                if mimic_window(poses, i, LEG_LENGTH[robot]).shape != (465,):
                    raise ValueError("Invalid WBC window")
            dest.mkdir(parents=True, exist_ok=True)
            outputs = {"poses.npy": poses.astype(np.float32), "joints.npy": joints.astype(np.float32)}
            for filename, array in outputs.items():
                temp = dest / (filename + ".tmp")
                with temp.open("wb") as stream:
                    np.save(stream, array, allow_pickle=False)
                os.replace(temp, dest / filename)
            saved = {"traj_id": f"{robot}/{name}", "source_id": row["source_id"],
                     "group_id": row.get("group_id", row["source_id"]), "split": row["split"],
                     "dataset": row["dataset"], "robot": robot, "frames": len(poses), "fps": 50,
                     "leg_length": LEG_LENGTH[robot], "path": str(relative), "input_sha256": expected,
                     "five_point_source": "paired_robot_fk", "point_names": [names[i] for i in indices],
                     "joint_names": data["joint_names"].tolist(),
                     "output_sha256": {f: sha256(dest/f) for f in outputs}}
        write_json(receipt_path, saved)
        return saved

    accepted, rejected = [], []
    def attempt(row):
        try:
            return convert(row), None
        except (ValueError, KeyError, FileNotFoundError) as error:
            return None, {"robot": row.get("robot"), "name": row.get("name"), "reason": str(error)}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, (saved, error) in enumerate(pool.map(attempt, document["clips"])):
            (accepted if saved else rejected).append(saved or error)
            if index % 500 == 0:
                print(f"Morph {index+1}/{len(document['clips'])}", flush=True)
    for robot in LEG_LENGTH:
        clips = [r for r in accepted if r["robot"] == robot]
        for row in clips:
            floor=floors[(robot,Path(row["path"]).name)]
            row.update(reference_floor_offset_m=floor["offset_m"],reference_floor_valid=floor["valid"])
        write_json(output_root / robot / "manifest.json", {"schema": "wammimic-morph-v1", "fps": 50,
                   "window_dim": 465, "normalization": "apply before mask zeroing", "clips": clips})
    report = {**identity, "complete": len(accepted)+len(rejected) == len(document["clips"]),
              "reference_floor_sha256":floor_sha,"reference_floor_unknown":sum(not r["reference_floor_valid"] for r in accepted),
              "accepted": len(accepted), "rejected": rejected,
              "robots": {robot:{"clips":sum(r["robot"]==robot for r in accepted),
                         "frames":sum(r["frames"] for r in accepted if r["robot"]==robot)} for robot in LEG_LENGTH},
              "physics_qa": "source FK reused; dynamic stability and collision checks not rerun"}
    write_json(output_root / "qa.json", report)
    return report
