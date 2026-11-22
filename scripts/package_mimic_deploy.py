"""Package verified Mimic ONNX models, representative motions and the full current library."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile

import numpy as np

ROBOTS = ("q1", "bumi", "g1")
COMPACT_ACTIONS = ("motiondecode_baji_WP_Baji_Quan_17851016", "dance1_subject1", "dance1_subject2")
PAYLOADS = (("reference_file", "reference_sha256"), ("human_file", "human_sha256"),
            ("five_file", "five_sha256"))


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def relative_file(root, path):
    path = Path(path)
    if not path.is_absolute():
        path = root / path
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"payload escapes source root: {path}")
    return str(PurePosixPath(root.name) / path.relative_to(root))


def make_index(native, five, native_root, five_root, manifest_sha):
    pairs = {(r["robot"], r["name"]): r for r in five["clips"]}
    if len(pairs) != len(five["clips"]):
        raise ValueError("duplicate five-point pair")
    rows = []
    seen = set()
    for row in native["clips"]:
        key = row["robot"], row["name"]
        if key in seen:
            raise ValueError(f"duplicate motion: {key}")
        seen.add(key)
        record = {k: row[k] for k in ("robot", "name", "dataset", "source_id", "split", "frames")}
        record.update(reference_file=relative_file(native_root, row["output"]),
                      reference_sha256=row["output_sha256"])
        if "fps" in row:
            record["fps"] = row["fps"]
        else:
            # MotionDecode manifest rows omit fps; the prepared NPZ owns the rate.
            with np.load(row["output"], allow_pickle=False) as payload:
                record["fps"] = int(payload["fps"])
        if record["fps"] != 50:
            raise ValueError(f"expected 50 Hz: {key}")
        if "phases" in row:
            record["phases"] = row["phases"]
        if row.get("human_available"):
            pair = pairs.pop(key, None)
            if pair is None:
                raise ValueError(f"missing authentic five-point pair: {key}")
            if (pair["frames"] != row["frames"] or pair["fps"] != 50
                    or pair["human_sha256"] != row["human_sha256"]
                    or pair["native_reference"]["sha256"] != row["output_sha256"]
                    or pair["input_manifest_sha256"] != manifest_sha):
                raise ValueError(f"five-point provenance differs: {key}")
            record.update(human_file=relative_file(native_root, row["human_sidecar"]),
                          human_sha256=row["human_sha256"],
                          five_file=relative_file(five_root, pair["output"]),
                          five_sha256=pair["output_sha256"])
        elif key in pairs:
            raise ValueError(f"motion-only source has a five-point pair: {key}")
        rows.append(record)
    if pairs:
        raise ValueError("five-point manifest contains motions outside the native library")
    return dict(training_manifest_sha256=manifest_sha, archive="motion-library.tar", clips=rows)


def expected_payloads(index):
    expected = {}
    for row in index["clips"]:
        for path_key, hash_key in PAYLOADS:
            if path_key in row:
                name, digest = row[path_key], row[hash_key]
                path = PurePosixPath(name)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError(f"unsafe indexed path: {name}")
                if name in expected and expected[name] != digest:
                    raise ValueError(f"conflicting payload hashes: {name}")
                expected[name] = digest
    return expected


def verify_archive(path, expected):
    seen, matched = {}, set()
    with tarfile.open(path, "r|*") as archive:
        for item in archive:
            if item.isfile():
                with archive.extractfile(item) as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
            elif item.islnk():
                if item.linkname not in seen:
                    raise ValueError(f"unresolved archive hard link: {item.name}")
                digest = seen[item.linkname]
            else:
                if item.name in expected:
                    raise ValueError(f"indexed payload is not a regular file: {item.name}")
                continue
            if item.name in seen:
                raise ValueError(f"duplicate archive member: {item.name}")
            seen[item.name] = digest
            if item.name in expected:
                if digest != expected[item.name]:
                    raise ValueError(f"archive payload hash mismatch: {item.name}")
                matched.add(item.name)
                if len(matched) % 10000 == 0:
                    print(f"verified {len(matched)}/{len(expected)} library payloads", flush=True)
    missing = set(expected) - matched
    if missing:
        raise ValueError(f"{len(missing)} indexed payloads missing: {sorted(missing)[:3]}")
    return dict(result="passed", verified_payloads=len(matched), archive_file_members=len(seen),
                scope="streaming SHA256 identity; no dynamics or hardware qualification")


def copy_checked(source, target, digest):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if sha256(target) != digest:
        raise ValueError(f"copied payload differs from current manifest: {source}")


def compact_selection(catalog, windows):
    recovery_files = {w["file"] for w in windows["windows"]}
    selected = [r for r in catalog["clips"] if r["name"] in COMPACT_ACTIONS or r["file"] in recovery_files]
    for robot in ROBOTS:
        rows = [r for r in selected if r["robot"] == robot]
        if len(rows) != 6 or {r["name"] for r in rows if r["name"] in COMPACT_ACTIONS} != set(COMPACT_ACTIONS):
            raise ValueError(f"expected three specified actions and three recovery clips: {robot}")
    return selected


def write_neutral(source, output, frame=11446):
    with np.load(source, allow_pickle=False) as z:
        count = len(z["joint_pos"])
        if not 0 <= frame < count:
            raise ValueError("neutral source frame is out of range")
        arrays = {k: np.repeat(z[k][frame:frame+1], 50, axis=0)
                  if z[k].ndim and z[k].shape[0] == count else z[k] for k in z.files}
    for key in ("joint_vel", "body_lin_vel_w", "body_ang_vel_w"):
        arrays[key] = np.zeros_like(arrays[key])
    np.savez_compressed(output, **arrays)


def check_representatives(root):
    catalog = json.loads((root / "references/catalog.json").read_text())
    metadata = {r: json.loads((root / f"model/{r}/metadata.json").read_text()) for r in ROBOTS}
    by_file = {r["file"]: r for r in catalog["clips"]}
    for row in catalog["clips"]:
        for file_key, hash_key in (("file", "sha256"), *PAYLOADS[1:]):
            if file_key in row and sha256(root / row[file_key]) != row[hash_key]:
                raise ValueError(f"representative hash mismatch: {row[file_key]}")
        with np.load(root / row["file"], allow_pickle=False) as z:
            if (int(z["fps"]) != 50 or len(z["joint_pos"]) != row["frames"]
                    or z["joint_names"].tolist() != metadata[row["robot"]]["native_joint_order"]):
                raise ValueError(f"reference format/joint order differs: {row['file']}")
            for key in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
                if len(z[key]) != row["frames"] or not np.isfinite(z[key]).all():
                    raise ValueError(f"nonfinite or inconsistent reference: {row['file']}/{key}")
            if row["name"] == "neutral":
                for key in ("joint_pos", "body_pos_w", "body_quat_w"):
                    np.testing.assert_array_equal(z[key], np.repeat(z[key][:1], row["frames"], axis=0))
                if any(np.count_nonzero(z[k]) for k in ("joint_vel", "body_lin_vel_w", "body_ang_vel_w")):
                    raise ValueError("neutral reference is not static")
        if "human_file" in row:
            with np.load(root / row["human_file"], allow_pickle=False) as z:
                if int(z["fps"]) != 50 or len(z["human_pos_w"]) != row["frames"]:
                    raise ValueError("human/reference frame mismatch")
                if not all(np.isfinite(z[k]).all() for k in ("human_pos_w", "human_quat_w")):
                    raise ValueError("nonfinite human reference")
            with np.load(root / row["five_file"], allow_pickle=False) as z:
                if z["human_pos_w"].shape != (row["frames"], 5, 3) or z["human_quat_w"].shape != (row["frames"], 5, 4):
                    raise ValueError("five-point/reference frame mismatch")
                if not all(np.isfinite(z[k]).all() for k in ("human_pos_w", "human_quat_w")):
                    raise ValueError("nonfinite five-point reference")
    windows = json.loads((root / "references/recovery_windows.json").read_text())
    if {(w["robot"], w["category"]) for w in windows["windows"]} != {
            (r, c) for r in ROBOTS for c in ("lying", "seated", "half_squat")}:
        raise ValueError("missing robot/recovery category")
    for window in windows["windows"]:
        row = by_file[window["file"]]
        source_file = f"references/{window['robot']}/approved-stand-source.npz"
        stand = by_file.get(source_file, by_file.get(f"references/{window['robot']}/neutral.npz"))
        target_frame = 11446 if source_file in by_file else 0
        if window["sha256"] != row["sha256"] or window["human_sha256"] != row["human_sha256"]:
            raise ValueError("recovery window/reference identity differs")
        with np.load(root / row["file"], allow_pickle=False) as z, np.load(root / stand["file"], allow_pickle=False) as target:
            start, hold = window["start_frame"], window["hold_frame"]
            if not 0 <= start < hold < row["frames"]:
                raise ValueError("invalid recovery frame range")
            for key in ("joint_pos", "body_pos_w", "body_quat_w"):
                np.testing.assert_array_equal(z[key][hold], z[key][0])
                np.testing.assert_allclose(z[key][hold], target[key][target_frame], atol=1e-6, rtol=0)
            if np.count_nonzero(z["joint_vel"][hold]):
                raise ValueError("standing hold has nonzero reference velocity")
    return dict(references=len(catalog["clips"]), human_pairs=sum("human_file" in r for r in catalog["clips"]),
                recovery_windows=len(windows["windows"]), result="passed")


def compact_package(source, output, guide):
    if output.exists():
        raise FileExistsError(output)
    catalog_file, windows_file = source / "references/catalog.json", source / "references/recovery_windows.json"
    catalog = json.loads(catalog_file.read_text())
    windows = json.loads(windows_file.read_text())
    selected = compact_selection(catalog, windows)
    sources = {str(p): sha256(p) for p in (catalog_file, windows_file, guide, Path(__file__).resolve())}
    model_manifest_file = source / "model/manifest.json"
    model_manifest = json.loads(model_manifest_file.read_text())
    sources[str(model_manifest_file)] = sha256(model_manifest_file)
    if model_manifest["iteration"] != 12000:
        raise ValueError("expected 12000-step models")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".mimic-compact-", dir=output.parent) as temp:
        stage = Path(temp)
        shutil.copytree(source / "model", stage / "model")
        for name, digest in model_manifest["files"].items():
            if sha256(stage / "model" / name) != digest:
                raise ValueError(f"model package hash mismatch: {name}")
        for row in selected:
            for file_key, hash_key in (("file", "sha256"), *PAYLOADS[1:]):
                if file_key in row:
                    copy_checked(source / row[file_key], stage / row[file_key], row[hash_key])
        for robot in ROBOTS:
            original = next(r for r in catalog["clips"] if r["robot"] == robot and r["name"] == "approved-stand-source")
            pose_source = source / original["file"]
            if sha256(pose_source) != original["sha256"]:
                raise ValueError("neutral source identity differs")
            sources[str(pose_source)] = original["sha256"]
            path = f"references/{robot}/neutral.npz"
            write_neutral(pose_source, stage / path)
            selected.append(dict(robot=robot, name="neutral", dataset="neutral_pose",
                                 source_id="lafan1/walk2_subject4@11446", split="derived_static",
                                 file=path, sha256=sha256(stage / path), fps=50, frames=50,
                                 human_available=False, source_sha256=original["sha256"],
                                 source_frame=11446, velocities="all zero"))
        catalog.update(clips=selected, source="specified three actions, three recovery poses and static neutral", model_iteration=12000)
        write_json(stage / "references/catalog.json", catalog)
        write_json(stage / "references/recovery_windows.json", windows)
        shutil.copytree(source / "licenses", stage / "licenses")
        shutil.copy2(guide, stage / "README_部署指南.md")
        shutil.copy2(Path(__file__), stage / "package_mimic_deploy.py")
        audit = check_representatives(stage)
        if audit["references"] != 21 or audit["human_pairs"] != 15 or audit["recovery_windows"] != 9:
            raise ValueError(f"unexpected compact package scope: {audit}")
        allowed = {row[k] for row in selected for k in ("file", "human_file", "five_file") if k in row}
        actual = {str(p.relative_to(stage)) for p in (stage / "references").rglob("*.npz")}
        if actual != allowed:
            raise ValueError("compact package contains missing or extra motion files")
        (stage / "validation").mkdir()
        audit.update(requested_actions=list(COMPACT_ACTIONS), neutral_reference_frames=50,
                     robot_references=21, human_sidecars=15, five_point_files=15,
                     motion_bytes=sum((stage / name).stat().st_size for name in allowed))
        write_json(stage / "validation/motion-audit.json", audit)
        for name, digest in sources.items():
            if sha256(name) != digest:
                raise ValueError(f"source changed during compact export: {name}")
        hashes = {str(p.relative_to(stage)): sha256(p) for p in sorted(stage.rglob("*")) if p.is_file()}
        (stage / "SHA256SUMS_ALL").write_text("".join(f"{digest}  {name}\n" for name, digest in hashes.items()))
        hashes["SHA256SUMS_ALL"] = sha256(stage / "SHA256SUMS_ALL")
        write_json(stage / "manifest.json", dict(schema_version=1, profile="compact", model_iteration=12000,
                   files=hashes, source_files=sources, representatives=audit, full_library_archive=None))
        for path in (stage, *[p for p in stage.rglob("*") if p.is_dir()]):
            path.chmod(0o755)
        if output.exists():
            raise FileExistsError(output)
        stage.rename(output)
    print(json.dumps(audit, indent=2), flush=True)
    print(f"Published compact package: {output}", flush=True)


def package(models, template, native_root, five_root, run, output, guide):
    if output.exists():
        raise FileExistsError(output)
    files = [native_root / "manifest.json", five_root / "manifest.json", run / "freeze.json",
             template / "references/catalog.json", template / "references/recovery_windows.json",
             native_root / "standing-review.json", native_root / "windows.json", Path(__file__).resolve()]
    sources = {str(p): sha256(p) for p in files}
    manifest_sha = sources[str(native_root / "manifest.json")]
    freeze = json.loads((run / "freeze.json").read_text())
    native, five = (json.loads((r / "manifest.json").read_text()) for r in (native_root, five_root))
    selection = json.loads((template / "references/catalog.json").read_text())
    if (not native["complete"] or not five["complete"] or freeze["data_manifest_sha256"] != manifest_sha
            or five["input_manifest_sha256"] != manifest_sha or selection["training_manifest_sha256"] != manifest_sha):
        raise ValueError("current training, five-point and template library identities differ")
    index = make_index(native, five, native_root, five_root, manifest_sha)
    expected = expected_payloads(index)
    by_key = {(r["robot"], r["name"]): r for r in native["clips"]}
    five_by_key = {(r["robot"], r["name"]): r for r in index["clips"] if "five_file" in r}
    standing = json.loads((native_root / "standing-review.json").read_text())
    model_manifest = json.loads((models / "manifest.json").read_text())
    if model_manifest["iteration"] != 12000:
        raise ValueError("expected the current 12000-step model package")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".mimic-delivery-", dir=output.parent) as temp:
        stage = Path(temp)
        shutil.copytree(models, stage / "model")
        for name, digest in model_manifest["files"].items():
            if sha256(stage / "model" / name) != digest:
                raise ValueError(f"model package integrity check failed: {name}")
        for robot in ROBOTS:
            metadata = json.loads((models / robot / "metadata.json").read_text())
            checkpoint = run / robot / "ckpt_12000.pt"
            if metadata["checkpoint"]["path"] != str(checkpoint):
                raise ValueError("model checkpoint belongs to another training run")
            if sha256(checkpoint) != metadata["checkpoint"]["sha256"]:
                raise ValueError("current checkpoint differs from the exported model")
            sources[str(checkpoint)] = metadata["checkpoint"]["sha256"]
        (stage / "references").mkdir()
        catalog = dict(training_manifest_sha256=manifest_sha, model_iteration=12000,
                       source="current training library; representative selection follows blinkmorph_20000", clips=[])
        for old in selection["clips"]:
            row = dict(old)
            key = row["robot"], row["name"]
            if row["name"] == "approved-stand-source":
                record = standing["robot_sources"][row["robot"]]
                source, digest = Path(record["npz"]), record["sha256"]
            else:
                record = by_key[key]
                source, digest = Path(record["output"]), record["output_sha256"]
                if row["frames"] != record["frames"] or row["split"] != record["split"]:
                    raise ValueError("representative selection differs from current library")
            if digest != row["sha256"]:
                raise ValueError("template reference is not the current data version")
            copy_checked(source, stage / row["file"], digest)
            if "human_file" in row:
                copy_checked(Path(record["human_sidecar"]), stage / row["human_file"], row["human_sha256"])
                pair = five_by_key[key]
                row["five_file"] = f"references/five_point/{row['robot']}/{row['name']}.npz"
                row["five_sha256"] = pair["five_sha256"]
                five_relative = PurePosixPath(pair["five_file"]).relative_to(five_root.name)
                copy_checked(five_root / str(five_relative), stage / row["five_file"], pair["five_sha256"])
            catalog["clips"].append(row)
        write_json(stage / "references/catalog.json", catalog)
        write_json(stage / "references/full-preview-index.json", index)
        shutil.copy2(template / "references/recovery_windows.json", stage / "references/recovery_windows.json")
        shutil.copytree(template / "licenses", stage / "licenses")
        (stage / "validation").mkdir()
        for name in ("standing-review.json", "in-place-validation.json", "windows.json"):
            shutil.copy2(native_root / name, stage / "validation" / name)
        shutil.copy2(run / "flip-windows.json", stage / "validation/flip-windows-12000.json")
        shutil.copy2(Path(__file__), stage / "package_mimic_deploy.py")
        shutil.copy2(guide, stage / "README_部署指南.md")
        representatives = check_representatives(stage)
        print(json.dumps(representatives), flush=True)
        counts = dict(robot_references=len(index["clips"]), human_sidecars=sum("human_file" in r for r in index["clips"]),
                      five_point_files=sum("five_file" in r for r in index["clips"]), indexed_payloads=len(expected),
                      by_robot=dict(Counter(r["robot"] for r in index["clips"])),
                      by_dataset=dict(Counter(r["dataset"] for r in index["clips"])))
        print("Archiving full current motion library: " + json.dumps(counts), flush=True)
        archive = stage / index["archive"]
        with tarfile.open(archive, "w", dereference=True) as tar:
            for data_root in (native_root, five_root):
                tar.add(data_root, arcname=data_root.name)
        audit = verify_archive(archive, expected)
        archive_sha = sha256(archive)
        audit.update(counts, archive=archive.name, archive_sha256=archive_sha,
                     archive_bytes=archive.stat().st_size, representatives=representatives,
                     training_manifest_sha256=manifest_sha)
        write_json(stage / "validation/library-audit.json", audit)
        for name, digest in sources.items():
            if sha256(name) != digest:
                raise ValueError(f"source changed while packaging: {name}")
        payload_hashes = {str(p.relative_to(stage)): (archive_sha if p == archive else sha256(p))
                          for p in sorted(stage.rglob("*")) if p.is_file()}
        (stage / "SHA256SUMS_ALL").write_text("".join(f"{v}  {k}\n" for k, v in payload_hashes.items()))
        payload_hashes["SHA256SUMS_ALL"] = sha256(stage / "SHA256SUMS_ALL")
        write_json(stage / "manifest.json", dict(schema_version=1, model_iteration=12000,
                   files=payload_hashes, source_files=sources, library=counts, representatives=representatives,
                   full_library_archive=index["archive"], model_source=str(models), template_selection=str(template)))
        if output.exists():
            raise FileExistsError(output)
        stage.rename(output)
    print(f"Published package: {output}", flush=True)


def check(root, full_archive=False):
    manifest = json.loads((root / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or sha256(path) != digest:
            raise ValueError(f"package hash mismatch: {name}")
    print(json.dumps(check_representatives(root)), flush=True)
    if full_archive:
        if not manifest.get("full_library_archive"):
            raise ValueError("this compact package has no full library archive")
        index = json.loads((root / "references/full-preview-index.json").read_text())
        print(json.dumps(verify_archive(root / index["archive"], expected_payloads(index))), flush=True)
    print("Package hashes and representative motion checks passed", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("models", "template", "data-root", "five-root", "run", "output", "guide", "check"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--compact-from", type=Path, help="Export only the specified actions, recovery and neutral from a previous package")
    parser.add_argument("--full-archive", action="store_true", help="with --check, rehash every indexed TAR payload")
    args = parser.parse_args()
    if args.check:
        check(args.check.resolve(strict=True), args.full_archive)
        return
    if args.compact_from:
        if not args.output or not args.guide:
            parser.error("--compact-from requires --output and --guide")
        compact_package(args.compact_from.absolute(), args.output.absolute(), args.guide.absolute())
        return
    required = (args.models, args.template, args.data_root, args.five_root, args.run, args.output, args.guide)
    if not all(required):
        parser.error("export requires --models --template --data-root --five-root --run --output --guide")
    models, template, native, five, run, output, guide = (p.absolute() for p in required)
    package(models, template, native, five, run, output, guide)


if __name__ == "__main__":
    main()
