import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from scripts.prepare_mimic_robot_five import SLOTS, reference_five


def test_named_fk_mapping_and_axis_round_trip():
    names = ['wrist_r', 'ankle_l', 'pelvis', 'wrist_l', 'ankle_r']
    native = Rotation.from_euler('xyz', np.arange(45).reshape(15, 3) / 40)
    motion = dict(fps=np.array(50), body_names=np.array(names), joint_pos=np.zeros((3, 2)),
                  body_pos_w=np.arange(45).reshape(3, 5, 3).astype(float),
                  body_quat_w=native.as_quat(scalar_first=True).reshape(3, 5, 4))
    ordered = ['pelvis', 'ankle_l', 'ankle_r', 'wrist_l', 'wrist_r']
    rest = Rotation.from_euler('z', np.pi/2) * Rotation.from_euler('x', np.pi/2)
    data, error = reference_five(motion, ordered, [str(i) for i in range(24)], rest)
    np.testing.assert_array_equal(data['human_pos_w'][:, SLOTS], motion['body_pos_w'][:, [2, 1, 4, 3, 0]])
    recovered = Rotation.from_quat(data['human_quat_w'][:, SLOTS].reshape(-1, 4), scalar_first=True) * rest.inv()
    expected = Rotation.from_quat(motion['body_quat_w'][:, [2, 1, 4, 3, 0]].reshape(-1, 4), scalar_first=True)
    assert np.max((recovered.inv() * expected).magnitude()) < 1e-6 and error < 1e-6
    assert np.flatnonzero(data['human_joint_valid']).tolist() == list(SLOTS)
    np.testing.assert_array_equal(data['source_time_s'], [0., .02, .04])
    motion['body_quat_w'][0, 0] = 0
    with pytest.raises(ValueError, match='Invalid reference'):
        reference_five(motion, ordered, [str(i) for i in range(24)], rest)
