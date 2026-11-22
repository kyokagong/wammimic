"""The optimizer must preserve actions and reject a different decoder layout."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
from onnx.reference import ReferenceEvaluator

spec = importlib.util.spec_from_file_location(
    "export_mimic_deploy", Path(__file__).parents[1] / "scripts/export_mimic_deploy.py")
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


def test_export_contract_matches_checkpoint_horizon():
    assert list(exporter.input_contract(1)) == ["motion", "human_h1"]
    assert list(exporter.input_contract(10)) == ["motion", "human_h10"]
    assert exporter.input_contract(10)["human_h10"]["ref_feedback"] == 160
    with pytest.raises(ValueError, match="horizon"):
        exporter.input_contract(2)


def test_validation_includes_robot_reference_without_claiming_human_labels():
    rows = [dict(robot="q1", name=str(i), split="val", human_available=True) for i in range(4)]
    rows.append(dict(robot="q1", name="baji", split="val", human_available=False,
                     five_point_available=True, five_point_source="robot_reference"))
    selected = exporter.validation_rows(rows, "q1")
    assert len(selected) == 4 and selected[-1]["name"] == "baji"
    assert not selected[-1]["human_available"]


def decoder_graph():
    h, n = onnx.helper, onnx.numpy_helper
    rng = np.random.default_rng(5)
    morph = rng.normal(size=154).astype("float32")
    constants = {"morph": morph[None]}
    nodes = []
    for branch in ("net", "reference_head.net"):
        prefix = "policy.decoder." + branch
        x = "x"
        for i in range(4 if branch == "net" else 1):
            name = f"{prefix}.{2*i}"
            width = 156 if i == 0 else 2
            if i == 0:
                nodes.append(h.make_node("Concat", [x, "morph"], [name + ".cat"], axis=1))
                x = name + ".cat"
            constants[name + ".weight"] = rng.normal(size=(2, width)).astype("float32")
            constants[name + ".bias"] = rng.normal(size=2).astype("float32")
            nodes.append(h.make_node("Gemm", [x, name + ".weight", name + ".bias"],
                                     [name + ".linear"], name=name, transB=1))
            x = name + ".linear"
            if branch == "net":
                constants[name + ".gamma"] = rng.normal(size=2).astype("float32")
                constants[name + ".beta"] = rng.normal(size=2).astype("float32")
                nodes.extend([h.make_node("Mul", [x, name + ".gamma"], [name + ".mul"]),
                              h.make_node("Add", [name + ".mul", name + ".beta"], [name + ".film"])])
                x = name + ".film"
        nodes.append(h.make_node("Identity", [x], [branch]))
    nodes.append(h.make_node("Add", ["net", "reference_head.net"], ["out"]))
    graph = h.make_graph(nodes, "decoder", [h.make_tensor_value_info("x", 1, [1, 2])],
                         [h.make_tensor_value_info("out", 1, [1, 2])],
                         [n.from_array(v, k) for k, v in constants.items()])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 18)])
    model.ir_version = 10
    return model, morph


def test_fold_preserves_both_paths():
    model, morph = decoder_graph()
    original = ReferenceEvaluator(onnx.ModelProto.FromString(model.SerializeToString()))
    report = exporter.optimize_graph(model, morph, output="out", action_dim=2)
    optimized = ReferenceEvaluator(model)
    assert report["film_layers"] == 4 and report["morph_concats"] == 2
    assert not any(n.op_type in ("Concat", "Mul") for n in model.graph.node)
    for x in (np.zeros((1, 2), np.float32), np.array([[0.3, -0.8]], np.float32)):
        np.testing.assert_allclose(original.run(None, {"x": x})[0],
                                   optimized.run(None, {"x": x})[0], atol=1e-4, rtol=0)


def test_unknown_layout_fails():
    model, morph = decoder_graph()
    model.graph.node[1].attribute[0].i = 0
    with pytest.raises(ValueError, match="Gemm"):
        exporter.optimize_graph(model, morph, output="out", action_dim=2)


def test_wrong_robot_morph_fails():
    model, morph = decoder_graph()
    with pytest.raises(ValueError, match="morph layout"):
        exporter.optimize_graph(model, morph + 1, output="out", action_dim=2)


def test_tampered_bundle_fails_before_inference(tmp_path):
    import json
    (tmp_path / "motion.onnx").write_bytes(b"changed")
    (tmp_path / "manifest.json").write_text(json.dumps({"files": {"motion.onnx": "0" * 64}}))
    with pytest.raises(ValueError, match="hash mismatch"):
        exporter.check_bundle(tmp_path)
