"""Convert one ROMP person (SMPL pose + camera) into the text payload read by Unreal Engine.

Payload body (the UDP sender appends "\\n"):

    q0x,q0y,q0z,q0w,...,q23x,q23y,q23z,q23w|c0,c1,c2

i.e. 24 joint quaternions (96 values) + 3 camera/translation values = 99 values,
each formatted with 4 decimals.
"""
import numpy as np
from scipy.spatial.transform import Rotation

# 180 degrees about the X axis, as an (x, y, z, w) quaternion.
ROOT_FLIP_X180_QUAT = np.array([1., 0., 0., 0.], dtype=np.float32)
# Scale applied to the camera values before sending (UE works in centimetres).
UE_TRANSLATION_SCALE = 100.0


def smpl_to_ue_pose(pose_rotvec, cam):
    """Return (joint_quats_xyzw (24, 4), translation (3,)) in UE conventions.

    pose_rotvec: (72,) float32 SMPL axis-angle pose (root joint first).
    cam:         (3,)  float32 ROMP camera parameters ``outputs['params']['cam'][i]``.
    """
    joint_quats = Rotation.from_rotvec(pose_rotvec.reshape(-1, 3)).as_quat()

    # SMPL (right-handed) -> UE (left-handed): mirror by negating the x and z components.
    joint_quats[:, 0] *= -1.0
    joint_quats[:, 2] *= -1.0
    # The SMPL root is upside down relative to the UE skeleton: rotate it 180 degrees about X.
    root = Rotation.from_quat(ROOT_FLIP_X180_QUAT) * Rotation.from_quat(joint_quats[0])
    joint_quats[0] = root.as_quat().astype(np.float32)

    translation = cam.copy()
    translation[1] *= -1.0  # flip to UE's axis direction
    translation[2] = 0.0    # fixed: the actor is kept on a constant plane in UE
    translation = translation * UE_TRANSLATION_SCALE
    return joint_quats, translation


def format_ue_pose_message(pose_rotvec, cam):
    """Return the payload body "<96 quaternion values>|<3 translation values>"."""
    joint_quats, translation = smpl_to_ue_pose(pose_rotvec, cam)
    pose_str = ",".join(f"{v:.4f}" for v in joint_quats.flatten())
    trans_str = ",".join(f"{v:.4f}" for v in translation.flatten())
    return f"{pose_str}|{trans_str}"
