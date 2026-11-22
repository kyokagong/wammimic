import hashlib
import importlib.util
import io
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "package_mimic_deploy", Path(__file__).parents[1] / "scripts/package_mimic_deploy.py")
package = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package)


class MotionPackageTests(unittest.TestCase):
    def test_compact_selection_contains_only_requested_actions_and_recovery(self):
        rows, windows = [], []
        for robot in package.ROBOTS:
            for name in (*package.COMPACT_ACTIONS, "unused_dance", "unused_flip", "approved-stand-source"):
                rows.append(dict(robot=robot, name=name, file=f"references/{robot}/{name}.npz"))
            for category in ("lying", "seated", "half_squat"):
                name = "recovery_" + category
                path = f"references/{robot}/{name}.npz"
                rows.append(dict(robot=robot, name=name, file=path))
                windows.append(dict(robot=robot, category=category, file=path))
        selected = package.compact_selection({"clips": rows}, {"windows": windows})
        self.assertEqual(len(selected), 18)
        self.assertFalse(any(r["name"].startswith("unused") for r in selected))
        self.assertFalse(any(r["name"] == "approved-stand-source" for r in selected))

    def test_neutral_reference_is_static_and_preserves_source_pose(self):
        with tempfile.TemporaryDirectory() as temp:
            source, target = Path(temp) / "source.npz", Path(temp) / "neutral.npz"
            np.savez(source, fps=50, joint_pos=np.arange(12).reshape(4, 3),
                     joint_vel=np.ones((4, 3)), body_pos_w=np.ones((4, 2, 3)),
                     body_quat_w=np.ones((4, 2, 4)), body_lin_vel_w=np.ones((4, 2, 3)),
                     body_ang_vel_w=np.ones((4, 2, 3)), joint_names=np.array(["a", "b", "c"]))
            package.write_neutral(source, target, frame=2)
            with np.load(target) as z:
                np.testing.assert_array_equal(z["joint_pos"], np.tile([6, 7, 8], (50, 1)))
                self.assertFalse(z["joint_vel"].any())
                self.assertFalse(z["body_lin_vel_w"].any())
                self.assertFalse(z["body_ang_vel_w"].any())
                self.assertEqual(z["joint_names"].tolist(), ["a", "b", "c"])

    def test_archive_payload_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp) / "library.tar"
            data = b"motion data"
            with tarfile.open(archive, "w") as tar:
                member = tarfile.TarInfo("native/q1/action.npz")
                member.size = len(data)
                tar.addfile(member, io.BytesIO(data))
            expected = {"native/q1/action.npz": hashlib.sha256(data).hexdigest()}
            result = package.verify_archive(archive, expected)
            self.assertEqual(result["verified_payloads"], 1)
            with self.assertRaisesRegex(ValueError, "mismatch"):
                package.verify_archive(archive, {"native/q1/action.npz": "0" * 64})

    def test_missing_archive_payload_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp) / "empty.tar"
            with tarfile.open(archive, "w"):
                pass
            with self.assertRaisesRegex(ValueError, "missing"):
                package.verify_archive(archive, {"native/q1/missing.npz": "0" * 64})

    def test_index_requires_authentic_five_point_pair(self):
        with tempfile.TemporaryDirectory() as temp:
            native, five = Path(temp) / "native", Path(temp) / "five"
            native.mkdir()
            five.mkdir()
            row = dict(robot="q1", name="walk", dataset="test", source_id="test/walk",
                       split="val", fps=50, frames=20, output=str(native / "q1/walk.npz"),
                       output_sha256="a" * 64, human_available=True,
                       human_sidecar=str(native / "human50/q1/walk.npz"), human_sha256="b" * 64)
            with self.assertRaisesRegex(ValueError, "five-point"):
                package.make_index({"clips": [row]}, {"clips": []}, native, five, "c" * 64)

    def test_motiondecode_fps_is_read_from_npz_when_manifest_omits_it(self):
        with tempfile.TemporaryDirectory() as temp:
            native, five = Path(temp) / "native", Path(temp) / "five"
            (native / "q1").mkdir(parents=True)
            five.mkdir()
            payload = native / "q1/baji.npz"
            row = dict(robot="q1", name="baji", dataset="motiondecode", source_id="test/baji",
                       split="train", frames=20, output=str(payload), output_sha256="a" * 64)
            np.savez(payload, fps=50)
            index = package.make_index({"clips": [row]}, {"clips": []}, native, five, "c" * 64)
            self.assertEqual(index["clips"][0]["fps"], 50)
            np.savez(payload, fps=120)
            with self.assertRaisesRegex(ValueError, "50 Hz"):
                package.make_index({"clips": [row]}, {"clips": []}, native, five, "c" * 64)


if __name__ == "__main__":
    unittest.main()
