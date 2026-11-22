"""Five-point geometry shared by WAM targets and Mimic windows."""
import numpy as np
from scipy.spatial.transform import Rotation

LEG_LENGTH = {"q1": 0.44, "bumi": 0.48, "g1": 0.68}
# Official OpenHLM OpenPI order -> G1 SDK order (legs, waist, arms).
OPENHLM_BODY = np.r_[16:28, 28:31, 0:7, 8:15]


def matrices(quaternions):
    q = np.asarray(quaternions, dtype=np.float64)
    if q.shape[-1] != 4 or not np.isfinite(q).all():
        raise ValueError("Expected finite wxyz quaternions")
    if not np.allclose(np.linalg.norm(q, axis=-1), 1, atol=1e-3):
        raise ValueError("Non-unit quaternion")
    return Rotation.from_quat(q.reshape(-1, 4)[:, [1, 2, 3, 0]]).as_matrix().reshape(q.shape[:-1] + (3, 3))


def rotation6d(six):
    """Inverse of the first-two-column 6D rotation representation."""
    x = np.asarray(six, dtype=np.float64)
    c1 = x[..., :3]
    c1 = c1 / np.clip(np.linalg.norm(c1, axis=-1, keepdims=True), 1e-8, None)
    c2 = x[..., 3:6]
    c2 = c2 - (c1 * c2).sum(axis=-1, keepdims=True) * c1
    c2 = c2 / np.clip(np.linalg.norm(c2, axis=-1, keepdims=True), 1e-8, None)
    c3 = np.cross(c1, c2)
    return np.stack((c1, c2, c3), axis=-1)


def features(poses, anchor, leg_length, *, full_rotation=False, floor_height=0.):
    """World xyz+wxyz -> 45D in one fixed observation reference frame.

    Lower-body positions use the support-foot ground projection; wrists use the
    pelvis origin. Rotations use heading for WAM, full pelvis attitude for WBC.
    """
    poses = np.asarray(poses)
    if poses.shape[-2:] != (5, 7) or not np.isfinite(poses).all():
        raise ValueError("Expected finite [...,5,7] five-point poses")
    if not np.isfinite(leg_length) or leg_length <= 0:
        raise ValueError("Invalid leg length")
    if not np.isfinite(floor_height):raise ValueError("Invalid reference floor")
    anchor = np.asarray(anchor)
    if anchor.shape != (5, 7) or not np.isfinite(anchor).all():
        raise ValueError("Anchor must contain all five world poses")
    pelvis = anchor[0]
    rotation = matrices(pelvis[3:])
    yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    heading = Rotation.from_euler("z", yaw).as_matrix()
    feet = anchor[1:3, :3]
    # Without force labels, near-equal ankle heights mean double support.
    support = feet[feet[:, 2] <= feet[:, 2].min() + 0.02, :2].mean(axis=0)
    position = (poses[..., :3] - pelvis[:3]) @ heading
    lower_origin = np.r_[support, float(floor_height)]
    position[..., :3, :] = (poses[..., :3, :3] - lower_origin) @ heading
    position[..., :3, :] /= leg_length
    relative = np.einsum("ij,...pjk->...pik", (rotation if full_rotation else heading).T, matrices(poses[..., 3:]))
    rot6d = relative[..., :, :2].swapaxes(-1, -2).reshape(poses.shape[:-1] + (6,))
    return np.concatenate([position.reshape(poses.shape[:-2] + (15,)), rot6d.reshape(poses.shape[:-2] + (30,))], axis=-1).astype(np.float32)


def mimic_window(poses, index, leg_length, point_mask=None, *, floor_height=0.):
    """Unnormalized 465D window; normalize 450D before applying masks in training."""
    if not 0 <= index < len(poses):
        raise IndexError(index)
    points = np.ones(5, dtype=bool) if point_mask is None else np.asarray(point_mask)
    if points.shape != (5,) or not np.isin(points, [0, 1]).all():
        raise ValueError("Expected binary five-point mask")
    count = min(10, len(poses) - index)
    block = np.zeros((10, 45), dtype=np.float32)
    block[:count] = features(poses[index:index+count], poses[index], leg_length, full_rotation=True,floor_height=floor_height)
    block[:, :15].reshape(10, 5, 3)[:, ~points.astype(bool)] = 0
    block[:, 15:].reshape(10, 5, 6)[:, ~points.astype(bool)] = 0
    return np.r_[block.ravel(), points.astype(np.float32), np.arange(10) < count].astype(np.float32)
