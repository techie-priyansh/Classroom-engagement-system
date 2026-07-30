import cv2
import mediapipe as mp
import numpy as np


class FaceMeshDetector:

    def __init__(
        self,
        max_num_faces=5,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ):

        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, roi):
        """
        Detect face landmarks on a cropped head ROI.

        Parameters
        ----------
        roi : ndarray
            Cropped BGR head image.

        Returns
        -------
        MediaPipe FaceLandmarks object or None.
        """

        if roi is None or roi.size == 0:
            return None

        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

        results = self.face_mesh.process(rgb)

        if results.multi_face_landmarks:
            return results.multi_face_landmarks[0]

        return None

    def get_landmark_points(self, landmarks, head_bbox):
        """
        Convert MediaPipe normalized landmarks into pixel
        coordinates of the ORIGINAL frame.

        Parameters
        ----------
        landmarks : MediaPipe FaceLandmarks
        head_bbox : (x1, y1, x2, y2)

        Returns
        -------
        numpy.ndarray
            Shape (N, 2) where N is usually 478.
        """

        if landmarks is None:
            return None

        x1, y1, x2, y2 = head_bbox

        roi_width = x2 - x1
        roi_height = y2 - y1

        landmark_points = []

        for landmark in landmarks.landmark:

            px = int(landmark.x * roi_width) + x1
            py = int(landmark.y * roi_height) + y1

            landmark_points.append([px, py])

        landmark_points = np.array(
            landmark_points,
            dtype=np.float32
        )

        return landmark_points

    def draw_landmarks(self, frame, landmark_points):
        """
        Draw landmarks on the original frame.

        Parameters
        ----------
        frame : ndarray
        landmark_points : ndarray (N,2)
        """

        if landmark_points is None:
            return

        for x, y in landmark_points:

            cv2.circle(
                frame,
                (int(x), int(y)),
                1,
                (0, 255, 0),
                -1,
            )