import cv2
import numpy as np


class HeadPoseEstimator:

    def __init__(self):

        self.model_points = np.array([
            (0.0, 0.0, 0.0),
            (0.0, -63.6, -12.5),
            (-43.3, 32.7, -26.0),
            (43.3, 32.7, -26.0),
            (-28.9, -28.9, -24.1),
            (28.9, -28.9, -24.1)
        ], dtype=np.float64)
    def rotation_matrix_to_euler_angles(self, R):
        """
            Convert a rotation matrix into Euler angles (yaw, pitch, roll).

            Returns angles in degrees.
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
            roll = 0

        pitch = np.degrees(pitch)
        yaw = np.degrees(yaw)
        roll = np.degrees(roll)

        return yaw, pitch, roll
    def estimate(self, landmark_points, frame_shape):

        image_points = np.array([
            landmark_points[1],
            landmark_points[152],
            landmark_points[33],
            landmark_points[263],
            landmark_points[61],
            landmark_points[291]
        ], dtype=np.float64)

        h, w = frame_shape[:2]

        focal_length = max(w,h)

        center = (w / 2, h / 2)

        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((5, 1), dtype=np.float64)

        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_SQPNP
        )

        if not success:
            return None

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        # projection_matrix = np.hstack((rotation_matrix, translation_vector))

        # _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(
        #     projection_matrix
        # )

        # pitch = float(euler_angles[0])
        # yaw = float(euler_angles[1])
        # roll = float(euler_angles[2])
        yaw, pitch, roll = self.rotation_matrix_to_euler_angles(
            rotation_matrix)

        return yaw, pitch, roll
