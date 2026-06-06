import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from threading import Lock


class PoseInterpolator:
    """Interpolates SMPL poses and root orientations.

    Quaternions are expected and returned in WXYZ (Hamilton) format.
    """

    def __init__(self):
        self.last_pose = None
        self.current_pose = None
        self.last_t = None
        self.current_t = None
        self._lock = Lock()

    def add_pose(
        self,
        timestamp: float,
        body_quat: np.ndarray,  # WXYZ format
        smpl_joints: np.ndarray,
        smpl_pose: np.ndarray,
        root_trans: np.ndarray = None,
    ):
        """Add a pose to the interpolator. body_quat must be in WXYZ format.

        Args:
            root_trans: optional (3,) root translation in camera space (metres).
                        Used for UE UDP/TCP streaming. Pass ``out["pred_cam_t"]``.
        """
        with self._lock:
            self.last_pose = self.current_pose
            self.last_t = self.current_t
            self.current_pose = (body_quat, smpl_joints, smpl_pose, root_trans)
            self.current_t = timestamp

    def get_latest_pose(self):
        """Returns the most recently added pose (body_quat in WXYZ)."""
        with self._lock:
            if self.current_pose is None:
                return None
            return self.current_pose[:3]

    def interpolate(self, query_t):
        with self._lock:
            if self.last_pose is None or self.current_pose is None:
                return None

            dt = self.current_t - self.last_t
            if dt <= 0:
                return None

            alpha = np.clip((query_t - self.last_t) / dt, 0.0, 1.0)

            last_quat, last_joints, last_pose, last_trans = self.last_pose
            curr_quat, curr_joints, curr_pose, curr_trans = self.current_pose

        # Slerp quaternion
        # Scipy uses XYZW, so we convert WXYZ -> XYZW for the calculation
        q_last = Rotation.from_quat(
            [last_quat[1], last_quat[2], last_quat[3], last_quat[0]]
        )
        q_curr = Rotation.from_quat(
            [curr_quat[1], curr_quat[2], curr_quat[3], curr_quat[0]]
        )
        slerp = Slerp([0, 1], Rotation.concatenate([q_last, q_curr]))
        q_interp = slerp([alpha])[0].as_quat()  # Returns XYZW

        # Convert back to WXYZ for internal consistency
        body_quat = np.array([q_interp[3], q_interp[0], q_interp[1], q_interp[2]])

        # Lerp joints
        smpl_joints = last_joints * (1 - alpha) + curr_joints * alpha

        # Slerp each joint pose
        smpl_pose = np.zeros_like(last_pose)
        for i in range(last_pose.shape[0]):
            r_last = Rotation.from_rotvec(last_pose[i])
            r_curr = Rotation.from_rotvec(curr_pose[i])
            slerp_i = Slerp([0, 1], Rotation.concatenate([r_last, r_curr]))
            smpl_pose[i] = slerp_i([alpha])[0].as_rotvec()

        # Lerp root translation (optional)
        root_trans = None
        if last_trans is not None and curr_trans is not None:
            root_trans = (
                np.asarray(last_trans, dtype=np.float64) * (1 - alpha)
                + np.asarray(curr_trans, dtype=np.float64) * alpha
            )

        return body_quat, smpl_joints, smpl_pose, root_trans
