import cv2
import numpy as np


class HeadPoseEstimator:
    """Head pose (yaw, pitch, roll in degrees) from 6 MediaPipe landmarks
    using solvePnP.

    Landmark points must be in FULL-FRAME pixel coordinates, because the
    camera matrix is built from the full frame size.
    """

    # MediaPipe landmark IDs, in the same order as self.model_points
    #   1 nose, 152 chin, 33 left eye, 263 right eye, 61 left mouth, 291 right mouth
    LANDMARK_IDS = [1, 152, 33, 263, 61, 291]

    def __init__(self):

        original_model = np.array([
            (0.0, 0.0, 0.0),          # Nose
            (0.0, -63.6, -12.5),      # Chin
            (-43.3, 32.7, -26.0),     # Left eye
            (43.3, 32.7, -26.0),      # Right eye
            (-28.9, -28.9, -24.1),    # Left mouth
            (28.9, -28.9, -24.1)      # Right mouth
        ], dtype=np.float64)

        # Convert the facial model (Y up, Z toward viewer) to the OpenCV
        # camera convention (Y down, Z away from camera).
        # This is a proper 180 degree rotation about X (det = +1).
        self.model_points = original_model * np.array(
            [1.0, -1.0, -1.0]
        )

        self._camera_shape = None
        self._camera_matrix = None
        self._dist_coeffs = np.zeros((5, 1), dtype=np.float64)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_camera_matrix(self, frame_shape):
        """Camera matrix is cached per frame size."""

        h, w = frame_shape[:2]

        if self._camera_shape != (h, w):

            focal_length = max(w, h)

            self._camera_matrix = np.array([
                [focal_length, 0, w / 2],
                [0, focal_length, h / 2],
                [0, 0, 1]
            ], dtype=np.float64)

            self._camera_shape = (h, w)

        return self._camera_matrix

    def rotation_matrix_to_euler_angles(self, R):
        """
        Convert a rotation matrix into Euler angles.

        Returns:
            yaw, pitch, roll in degrees
        """

        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)

        singular = sy < 1e-6

        if not singular:

            pitch = np.arctan2(R[2, 1], R[2, 2])
            yaw = np.arctan2(-R[2, 0], sy)
            roll = np.arctan2(R[1, 0], R[0, 0])

        else:

            pitch = np.arctan2(-R[1, 2], R[1, 1])
            yaw = np.arctan2(-R[2, 0], sy)
            roll = 0.0

        return (
            float(np.degrees(yaw)),
            float(np.degrees(pitch)),
            float(np.degrees(roll)),
        )

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def estimate_full(self, landmark_points, frame_shape):
        """
        Returns a dict with yaw, pitch, roll (degrees) plus rvec, tvec,
        camera_matrix and dist_coeffs (handy for drawing axes),
        or None if the pose could not be solved.
        """

        image_points = np.array(
            [landmark_points[i][:2] for i in self.LANDMARK_IDS],
            dtype=np.float64
        )

        camera_matrix = self.get_camera_matrix(frame_shape)

        # Start the solver from a sensible guess: a frontal face, in front
        # of the camera, at the distance implied by the eye spacing.
        # Without a guess, the iterative solver sometimes converges to a
        # mirrored solution that lies BEHIND the camera, and those frames
        # were thrown away (6-24 % of faces on the test clip).
        focal = camera_matrix[0, 0]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

        # model eye corners (ids 33 and 263) are 86.6 mm apart
        eye_px = float(np.linalg.norm(image_points[2] - image_points[3]))

        if eye_px < 1.0:
            return None

        tz = focal * 86.6 / eye_px
        tx = (image_points[0][0] - cx) * tz / focal
        ty = (image_points[0][1] - cy) * tz / focal

        rvec = np.zeros((3, 1), dtype=np.float64)
        tvec = np.array([[tx], [ty], [tz]], dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            self.model_points,
            image_points,
            camera_matrix,
            self._dist_coeffs,
            rvec,
            tvec,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return None
        
        R, _ = cv2.Rodrigues(rvec)

        camera_points = (
            R @ self.model_points.T + tvec
        ).T

        if np.any(camera_points[:, 2] <= 0):
            return None
        # The face must be in front of the camera
        # if tvec[2, 0] <= 0:
        #     return None

        projected_points, _ = cv2.projectPoints(
            self.model_points,
            rvec,
            tvec,
            camera_matrix,
            self._dist_coeffs
        )

        reprojection_error = np.mean(
            np.linalg.norm(
                image_points - projected_points.reshape(-1, 2),
                axis=1
            )
        )
       

        yaw, pitch, roll = self.rotation_matrix_to_euler_angles(R)

        return {
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
            "reprojection_error": reprojection_error,
            "rvec": rvec,
            "tvec": tvec,
            "camera_matrix": camera_matrix,
            "dist_coeffs": self._dist_coeffs,
        }

    def estimate(self, landmark_points, frame_shape):
        """Backward-compatible: returns (yaw, pitch, roll) or None."""

        result = self.estimate_full(landmark_points, frame_shape)

        if result is None:
            return None

        return result["yaw"], result["pitch"], result["roll"]


class AngleSmoother:
    """Per-student exponential moving average of (yaw, pitch, roll).

    alpha close to 1 -> follows the raw value, close to 0 -> very smooth.
    """

    def __init__(self, alpha=0.4):
        self.alpha = alpha
        self._state = {}

    def update(self, key, angles):

        angles = np.asarray(angles, dtype=np.float64)
        previous = self._state.get(key)

        if previous is None:
            smoothed = angles
        else:
            smoothed = self.alpha * angles + (1.0 - self.alpha) * previous

        self._state[key] = smoothed

        return tuple(float(a) for a in smoothed)
    
    def reset(self):
        """Clear previous head-pose measurements."""
        self._state.clear()
        print("AngleSmoother reset successfully.")