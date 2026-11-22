"""Export fixed-robot Global + Ref. Mimic motion and checkpoint-matched five-point ONNX.

Run with the frozen training source and its robot assets in the Python environment.
Only trusted, numbered local checkpoints are accepted. See docs/train/mimic.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time

import numpy as np

ATOL = 1e-4
POINTS = [0, 7, 8, 20, 21]
INPUTS = {
    "motion": {"obs_history": 930, "goal_motion": 670, "ref_feedback": 160},
    "human_h1": {"obs_history": 930, "goal_human": 2160,
                 "human_heading": 60, "ref_feedback": 160},
}


def input_contract(horizon):
    if horizon not in (1, 10):
        raise ValueError(f"unsupported human horizon: {horizon}")
    return {"motion": INPUTS["motion"], f"human_h{horizon}": INPUTS["human_h1"]}


def validation_rows(rows, robot):
    eligible = [r for r in rows if r["robot"] == robot and r["split"] == "val"
                and r.get("five_point_available", r.get("human_available", False))]
    selected = eligible[:4]
    if len(selected) != 4:
        raise ValueError("need four paired validation clips")
    reference = next((r for r in eligible if r.get("five_point_source") == "robot_reference"), None)
    if reference is not None and reference not in selected:
        selected[-1] = reference
    return selected


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, doc):
    Path(path).write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")


def optimize_graph(model, morph, output, action_dim=29):
    """Fold only the pinned decoder layout; fail closed on structural changes."""
    import onnx
    from onnx import helper, numpy_helper as nh
    from onnx.reference import ReferenceEvaluator

    graph = model.graph
    before = len(graph.node)
    constants = {t.name: nh.to_array(t).copy() for t in graph.initializer}
    # Evaluate constant-only morphology/statistics chains before pattern matching.
    kept = []
    for node in graph.node:
        if all(not k or k in constants for k in node.input):
            values = ReferenceEvaluator(node, opsets={o.domain: o.version for o in model.opset_import}).run(
                None, {k: constants[k] for k in node.input if k})
            constants.update(zip(node.output, values, strict=True))
        else:
            kept.append(node)
    del graph.node[:]
    graph.node.extend(kept)

    def put(name, value):
        constants[name] = np.ascontiguousarray(value, dtype=np.float32)

    def gemm_parameters(node):
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        if (node.op_type != "Gemm" or len(node.input) != 3 or attrs.get("transB", 0) != 1
                or attrs.get("transA", 0) != 0 or attrs.get("alpha", 1.) != 1.
                or attrs.get("beta", 1.) != 1. or any(k not in constants for k in node.input[1:])):
            raise ValueError(f"unsupported Gemm: {node.name}")
        weight, bias = (constants[k] for k in node.input[1:])
        if weight.ndim != 2 or bias.shape != (weight.shape[0],):
            raise ValueError(f"unsupported Gemm parameter shapes: {node.name}")
        return weight, bias

    def only_consumer(name, op):
        found = [n for n in graph.node if name in n.input]
        if len(found) != 1 or found[0].op_type != op:
            raise ValueError(f"expected one {op} consumer of {name}")
        return found[0]

    removed = set()
    film = 0
    concats = 0
    for node in list(graph.node):
        if node.op_type != "Gemm":
            continue
        name = node.input[1]
        if re.search(r"(?:^|\.)decoder\.net\.(0|2|4|6)\.weight$", name):
            w, b = gemm_parameters(node)
            mul = only_consumer(node.output[0], "Mul")
            add = only_consumer(mul.output[0], "Add")
            gamma_name = next(k for k in mul.input if k != node.output[0])
            beta_name = next(k for k in add.input if k != mul.output[0])
            if gamma_name not in constants or beta_name not in constants:
                raise ValueError(f"FiLM is not constant: {name}")
            gamma, beta = constants[gamma_name].reshape(-1), constants[beta_name].reshape(-1)
            if gamma.shape != b.shape or beta.shape != b.shape:
                raise ValueError(f"FiLM width differs: {name}")
            put(node.input[1], gamma[:, None] * w)
            put(node.input[2], gamma * b + beta)
            node.output[0] = add.output[0]
            removed.update((mul.name or mul.output[0], add.name or add.output[0]))
            film += 1
        if re.search(r"(?:^|\.)decoder\.(?:reference_head\.)?net\.0\.weight$", name):
            w, b = gemm_parameters(node)
            producers = [n for n in graph.node if node.input[0] in n.output]
            if len(producers) != 1 or producers[0].op_type != "Concat":
                raise ValueError(f"expected morph Concat before {name}")
            cat = producers[0]
            axis = next((a.i for a in cat.attribute if a.name == "axis"), None)
            value = constants.get(cat.input[-1])
            if (axis not in (1, -1) or value is None or value.shape != (1, len(morph))
                    or not np.array_equal(value[0], morph) or len(cat.input) < 2
                    or w.shape[1] <= len(morph)):
                raise ValueError(f"unsupported trailing morph layout: {name}")
            only_consumer(cat.output[0], "Gemm")
            put(node.input[2], b + w[:, -len(morph):] @ morph)
            put(node.input[1], w[:, :-len(morph)])
            del cat.input[-1]
            if len(cat.input) == 1:
                node.input[0] = cat.input[0]
                removed.add(cat.name or cat.output[0])
            concats += 1
    if (film, concats) != (4, 2):
        raise ValueError(f"expected four FiLM layers and two morph concats, got {film}, {concats}")
    kept = [n for n in graph.node if (n.name or n.output[0]) not in removed]
    del graph.node[:]
    graph.node.extend(kept)
    del graph.initializer[:]
    graph.initializer.extend(nh.from_array(np.asarray(v), k) for k, v in constants.items())
    cleaned = onnx.utils.Extractor(model).extract_model([i.name for i in graph.input], [output])
    model.CopyFrom(cleaned)
    # Stale value_info shapes refer to the wider pre-fold Concat.
    del model.graph.value_info[:]
    onnx.checker.check_model(model, full_check=True)
    if list(model.graph.output[0].type.tensor_type.shape.dim)[-1].dim_value != action_dim:
        raise ValueError("wrong action width")
    return dict(film_layers=film, morph_concats=concats,
                nodes_before=before, nodes_after=len(model.graph.node))


def load_actor(checkpoint):
    import torch
    from torch import nn
    from deploy.export_onnx import _build_policy
    from morphmimic.constants import GOAL_VERSION
    from morphmimic.model.human import fixed_human_masks, mask_human_goal
    from morphmimic.robots import check_imu_body, get_robot
    from morphmimic.robots.morphology import morphology_vector
    from morphmimic.train.normalizers import RunningNormalizer, normalize_history

    match = re.fullmatch(r"ckpt_(\d+)\.pt", checkpoint.name)
    if checkpoint.is_symlink() or match is None:
        raise ValueError("use a trusted numbered ckpt_<iteration>.pt, not a symlink")
    digest = sha256(checkpoint)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    if (int(match[1]) != ckpt["iteration"] or ckpt.get("goal_version") != GOAL_VERSION
            or ckpt.get("kp_goal_version") != 10 or cfg.get("goal_anchor") != "reference"
            or cfg.get("policy_goal", "motion") != "motion" or cfg["hist_len"] != 10
            or not cfg.get("ref_feedback") or not cfg.get("ref_reconstruction_skip")):
        raise ValueError("expected the motion+human Global + Ref. checkpoint contract")
    if "ema_policy" not in ckpt:
        raise ValueError("EMA policy required; no online fallback")
    check_imu_body(ckpt, checkpoint)
    state = ckpt["ema_policy"]
    if any(not torch.isfinite(v).all() for v in state.values()):
        raise ValueError("nonfinite policy parameters")
    spec = get_robot(cfg["robot"])
    morph = state["decoder.morph_actor"]
    current = morphology_vector(spec.name, actor=True, actor_geometry=True)
    if (morph.shape != (154,) or not torch.equal(morph, ckpt["morph"]["actor"])
            or not torch.allclose(morph, current, atol=1e-6, rtol=0)
            or not torch.equal(state["decoder.joint_mask"], torch.tensor(spec.translator.joint_mask))
            or not torch.equal(morph[-29:], state["decoder.joint_mask"])):
        raise ValueError("checkpoint morphology/mask differs from robot assets")

    class Actor(nn.Module):
        def __init__(self):
            super().__init__()
            self.horizon = int(cfg.get("kp_fixed_horizon") or 1)
            input_contract(self.horizon)
            self.policy = _build_policy(ckpt)
            for key, dim in (("obs", 93), ("goal", 670), ("human", 2160), ("heading", 60)):
                norm = RunningNormalizer(dim)
                norm.load_state_dict(ckpt[key + "_norm"])
                if (norm.mean.shape != (dim,) or norm.var.shape != (dim,)
                        or not torch.isfinite(norm.mean).all() or not torch.isfinite(norm.var).all()
                        or (norm.var < 0).any()):
                    raise ValueError(f"invalid {key} normalizer")
                setattr(self, key + "_norm", norm)
            for key, value in zip(("points", "frames", "heading_mask"),
                                  fixed_human_masks(1, "five", self.horizon, "cpu"), strict=True):
                self.register_buffer(key, value)

        def forward(self, obs_history, goal_motion, ref_feedback):
            obs = normalize_history(self.obs_norm, obs_history)
            return self.policy(obs, self.goal_norm(goal_motion), ref_feedback=ref_feedback)

        def human(self, obs_history, goal_human, human_heading, ref_feedback):
            obs = normalize_history(self.obs_norm, obs_history)
            batch = obs_history.shape[0]
            goal = mask_human_goal(self.human_norm(goal_human), self.points.expand(batch, -1),
                                   self.frames.expand(batch, -1), self.heading_norm(human_heading),
                                   self.heading_mask.expand(batch))
            return self.policy.forward_human(obs, goal, ref_feedback=ref_feedback)

    actor = Actor().eval()
    if sha256(checkpoint) != digest:
        raise ValueError("checkpoint changed while loading")
    ckpt["_checkpoint_sha256"] = digest
    return actor, ckpt


def reference_inputs(actor, ckpt, data_root):
    """Four held-out paired clips, kinematic histories; no physics/reward claim."""
    import torch
    from morphmimic.constants import ANCHOR_IDX
    from morphmimic.data.motion_library import MotionLibrary
    from morphmimic.model.human import human_features_from_goal, heading_from_goal, pack_human_goal
    from morphmimic.model.ref_feedback import global_ref_feedback
    from morphmimic.model.rewards_torch import analytic_obs, goal_features
    from morphmimic.model.state import split_state
    from morphmimic.robots import get_robot

    manifest = data_root / "manifest.json"
    manifest_hash = sha256(manifest)
    rows = json.loads(manifest.read_text())["clips"]
    spec = get_robot(ckpt["cfg"]["robot"])
    rows = validation_rows(rows, spec.name)
    human_root = Path(ckpt["cfg"].get("human_root") or data_root / "human50")
    files = {str(manifest): manifest_hash}
    samples = []
    origins = []
    for i, row in enumerate(rows):
        side_key, hash_key = (("five_point_sidecar", "five_point_sha256") if row.get("five_point_available")
                              else ("human_sidecar", "human_sha256"))
        for key, digest_key in (("output", "output_sha256"), (side_key, hash_key)):
            path = Path(row[key])
            digest = sha256(path)
            if digest != row[digest_key]:
                raise ValueError(f"validation source changed: {path}")
            files[str(path)] = digest
        effective_sidecar = human_root / spec.name / Path(row["output"]).name
        if sha256(effective_sidecar) != row[hash_key]:
            raise ValueError("training sidecar differs from manifest label source")
        files[str(effective_sidecar)] = row[hash_key]
        lib = MotionLibrary([row["output"]], robot=spec, human_root=str(human_root))
        frame = max(10, min(int(lib.clip_len[0]) - 11, int(lib.clip_len[0]) * (i + 1) // 5))
        f = torch.tensor([frame])
        if not lib.human_valid_at(f)[0, POINTS].all():
            raise ValueError("fixture is missing a required five-point slot")
        state, window = lib.state(f), lib.future(f, 10)
        history = lib.state(torch.arange(frame - 9, frame + 1)[None])
        zeros = torch.zeros(1, 10, 29)
        obs = analytic_obs(history, zeros, ckpt["wm"]["default_joint_pos"], zeros,
                           imu_idx=spec.imu_slot, joint_mask=actor.policy.decoder.joint_mask).flatten(1)
        human = lib.human_future(f, 10)
        raw = pack_human_goal(human[..., :3], human[..., 3:], split_state(state).body_quat[:, ANCHOR_IDX])
        samples.append(dict(obs_history=obs, goal_motion=goal_features(state, window),
                            goal_human=human_features_from_goal(raw, actor.points),
                            human_heading=heading_from_goal(raw, actor.points),
                            ref_feedback=global_ref_feedback(state, window[:, :4], lib.remaining(f)),
                            raw_human=raw))
        origins.append(dict(clip=row["name"], frame=frame, split="val",
                            five_point_source=row.get("five_point_source", "original_human")))
    arrays = {k: torch.cat([s[k] for s in samples]).numpy() for k in samples[0]}
    return arrays, dict(kind="kinematic reference histories; zero previous actions; no physics rollout",
                        samples=origins, source_files=files)


def session(path):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def verify_graph(path, original_path, arrays, names, expected, mask):
    optimized, original = session(path), session(original_path)
    errors = {"optimized_vs_original_onnx": 0., "optimized_vs_pytorch": 0., "original_onnx_vs_pytorch": 0.}
    for i in range(len(expected)):
        feed = {k: arrays[k][i:i+1] for k in names}
        got = optimized.run(None, feed)[0]
        before = original.run(None, feed)[0]
        want = expected[i:i+1]
        for value in (got, before, want):
            if value.shape != (1, 29) or not np.isfinite(value).all() or np.count_nonzero(value[:, ~mask]):
                raise ValueError(f"{path}: nonfinite/wrong shape/unmasked missing joints, sample {i}")
        for key, a, b in (("optimized_vs_original_onnx", got, before),
                          ("optimized_vs_pytorch", got, want), ("original_onnx_vs_pytorch", before, want)):
            errors[key] = max(errors[key], float(np.max(np.abs(a - b))))
    if max(errors.values()) > ATOL:
        raise ValueError(f"{path}: parity exceeds {ATOL}: {errors}")
    # Preserve the trained invalid-current-feedback rejection (NaN actions).
    invalid = {k: arrays[k][0:1].copy() for k in names}
    invalid["ref_feedback"][0, 144] = 0
    if not np.isnan(optimized.run(None, invalid)[0]).all():
        raise ValueError(f"{path}: invalid feedback guard was lost")
    return dict(cases=len(expected), max_abs_error=errors, absolute_tolerance=ATOL,
                invalid_feedback_guard="passed", absent_actions_exactly_zero=True)


def export(checkpoint, source, data_root, output):
    import torch
    import onnx
    import onnxruntime as ort
    from torch import nn
    from morphmimic.scripts.export import _metadata
    from morphmimic.train.imagination import act, act_human

    if output.exists():
        raise FileExistsError(output)
    source_files = sorted((source / "src/morphmimic").rglob("*.py")) + [source / "deploy/export_onnx.py", Path(__file__).resolve()]
    sources = {str(p): sha256(p) for p in source_files}
    sources[str(checkpoint)] = sha256(checkpoint)
    actor, ckpt = load_actor(checkpoint)
    contracts = input_contract(actor.horizon)
    metadata = _metadata(actor, ckpt, checkpoint, "ema")
    base, provenance = reference_inputs(actor, ckpt, data_root)
    sources.update(provenance["source_files"])
    with torch.inference_mode():
        h, g, f = (torch.from_numpy(base[k]) for k in ("obs_history", "goal_motion", "ref_feedback"))
        motion = act(actor.policy, actor.obs_norm, actor.goal_norm, h, g, f).numpy()
        human = act_human(actor.policy, actor.obs_norm, actor.human_norm, actor.heading_norm,
                          h, torch.from_numpy(base["raw_human"]),
                          actor.points.expand(4, -1), actor.frames.expand(4, -1),
                          actor.heading_mask.expand(4), f).numpy()
        direct = actor.human(h, torch.from_numpy(base["goal_human"]), torch.from_numpy(base["human_heading"]), f).numpy()
    if not np.allclose(direct, human, atol=1e-6, rtol=0):
        raise ValueError("human feature boundary differs from native act_human")
    rng = np.random.default_rng(0)
    arrays = {}
    for key in dict.fromkeys(k for widths in contracts.values() for k in widths):
        perturb = base[key][np.arange(256) % 4] + rng.normal(0, .2, (256, base[key].shape[1])).astype("float32")
        if key == "ref_feedback":
            perturb[:, 144:] = base[key][np.arange(256) % 4, 144:]
        arrays[key] = np.concatenate([base[key], perturb])
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".mimic-export-", dir=output.parent) as temp:
        stage = Path(temp)
        verification = stage / "verification"
        verification.mkdir()
        reports = {}
        mask = np.array(metadata["joint_mask"], dtype=bool)

        class Human(nn.Module):
            def __init__(self, actor):
                super().__init__()
                self.actor = actor

            def forward(self, obs_history, goal_human, human_heading, ref_feedback):
                return self.actor.human(obs_history, goal_human, human_heading, ref_feedback)

        for branch, widths in contracts.items():
            wrapper = actor if branch == "motion" else Human(actor).eval()
            names = list(widths)
            output_name = "action_motion" if branch == "motion" else "action_human"
            original = verification / (branch + ".original.onnx")
            args = tuple(torch.from_numpy(base[k][0:1]) for k in names)
            with torch.inference_mode():
                torch.onnx.export(wrapper, args, str(original), opset_version=18, dynamo=False,
                                  input_names=names, output_names=[output_name], do_constant_folding=True)
                expected = np.concatenate([wrapper(*(torch.from_numpy(arrays[k][i:i+1]) for k in names)).numpy()
                                           for i in range(260)])
            native = motion if branch == "motion" else human
            native_error = float(np.max(np.abs(expected[:4] - native)))
            if not np.isfinite(native_error) or native_error > ATOL:
                raise ValueError(f"single-frame wrapper differs from native batched inference: {native_error}")
            arrays[branch + "_expected"] = expected
            model = onnx.load(original)
            onnx.checker.check_model(model, full_check=True)
            report = optimize_graph(model, actor.policy.decoder.morph_actor.numpy(), output_name)
            report["native_base_max_abs_error"] = native_error
            graph = stage / (branch + ".onnx")
            onnx.save(model, graph)
            actual = {x.name: [d.dim_value for d in x.type.tensor_type.shape.dim] for x in model.graph.input}
            if actual != {k: [1, d] for k, d in widths.items()}:
                raise ValueError(f"unexpected runtime input contract: {actual}")
            report.update(verify_graph(graph, original, arrays, names, expected, mask))
            report.update(inputs=actual, output={output_name: [1, 29]}, bytes=graph.stat().st_size,
                          original_bytes=original.stat().st_size)
            reports[branch] = report
            print(json.dumps(dict(robot=metadata["robot"], branch=branch, validation=report)), flush=True)
        np.savez_compressed(verification / "inputs.npz", **arrays)
        metadata.update(schema_version=2, inputs={k: r["inputs"] for k, r in reports.items()},
                        output={k: r["output"] for k, r in reports.items()},
                        batch_size=1, ref_feedback_contract=ckpt["ref_feedback"],
                        human_horizon=actor.horizon,
                        human_masks=dict(points=POINTS, frames=list(range(actor.horizon)), heading=True),
                        normalizers={k: dict(mean=getattr(actor, k + "_norm").mean.tolist(),
                                             var=getattr(actor, k + "_norm").var.tolist(), eps=.01)
                                     for k in ("obs", "goal", "human", "heading")},
                        validation=reports, validation_inputs=provenance,
                        perturbation=dict(seed=0, samples=256, raw_feature_std=.2, validity_bits_unchanged=True),
                        versions=dict(torch=torch.__version__, onnx=onnx.__version__, onnxruntime=ort.__version__),
                        precision="float32; no quantization", source_files=sources)
        metadata.pop("parity", None)
        write_json(stage / "metadata.json", metadata)
        shutil.copy2(Path(__file__), stage / "export_mimic_deploy.py")
        files = {str(p.relative_to(stage)): sha256(p) for p in sorted(stage.rglob("*")) if p.is_file()}
        (stage / "SHA256SUMS").write_text("".join(f"{v}  {k}\n" for k, v in files.items()))
        for path, digest in sources.items():
            if sha256(path) != digest:
                raise ValueError(f"source changed during export: {path}")
        # The destination is a new directory; the manifest is the final staged file.
        write_json(stage / "manifest.json", dict(schema_version=1, files=files,
                                                 source_files=sources, validation=reports))
        if output.exists():
            raise FileExistsError(output)
        stage.rename(output)
    return metadata


def check_bundle(bundle, benchmark=0):
    manifest = json.loads((bundle / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        path = (bundle / name).resolve()
        if not path.is_relative_to(bundle.resolve()) or sha256(path) != digest:
            raise ValueError(f"artifact hash mismatch: {name}")
    metadata = json.loads((bundle / "metadata.json").read_text())
    mask = np.array(metadata["joint_mask"], dtype=bool)
    report = {}
    with np.load(bundle / "verification/inputs.npz", allow_pickle=False) as arrays:
        for branch, names in metadata["inputs"].items():
            graph, original = bundle / (branch + ".onnx"), bundle / "verification" / (branch + ".original.onnx")
            report[branch] = verify_graph(graph, original, arrays, names, arrays[branch + "_expected"], mask)
            if benchmark:
                timings = {}
                feed = {k: arrays[k][0:1] for k in names}
                for label, path in (("original", original), ("optimized", graph)):
                    runtime = session(path)
                    for _ in range(30):
                        runtime.run(None, feed)
                    measured = []
                    for _ in range(benchmark):
                        start = time.perf_counter_ns()
                        runtime.run(None, feed)
                        measured.append((time.perf_counter_ns() - start) / 1e6)
                    p50, p95, p99 = np.percentile(measured, [50, 95, 99])
                    timings[label] = dict(samples=benchmark, p50_ms=p50, p95_ms=p95, p99_ms=p99,
                                          max_ms=max(measured), over_20ms=sum(t > 20 for t in measured))
                timings["p50_speedup"] = timings["original"]["p50_ms"] / timings["optimized"]["p50_ms"]
                report[branch]["host_cpu_benchmark"] = timings
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Frozen BlinkMorphMimic source directory")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", type=Path, help="Independently recheck a published bundle (no torch/assets needed)")
    parser.add_argument("--benchmark", type=int, default=0, help="CPU timing repetitions with --check")
    args = parser.parse_args()
    if args.benchmark < 0:
        parser.error("--benchmark must be nonnegative")
    if args.check:
        check_bundle(args.check.resolve(), args.benchmark)
        return
    if not all((args.source, args.checkpoint, args.data_root, args.output)):
        parser.error("export requires --source, --checkpoint, --data-root and --output")
    source = args.source.resolve(strict=True)
    sys.path[:0] = [str(source / "src"), str(source)]
    import morphmimic
    if not Path(morphmimic.__file__).resolve().is_relative_to(source):
        raise ValueError("morphmimic was already imported from a different source")
    import torch
    torch.set_num_threads(1)
    export(args.checkpoint.absolute(), source, args.data_root.resolve(strict=True), args.output.absolute())


if __name__ == "__main__":
    main()
