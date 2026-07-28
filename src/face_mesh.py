import cv2
import mediapipe as mp


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
        Detect face landmarks on a cropped ROI.

        Args:
            roi : BGR image

        Returns:
            MediaPipe landmarks or None
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
        Convert MediaPipe normalized landmarks into
        pixel coordinates on the original frame.

        Args:
            landmarks: MediaPipe landmark object
            head_bbox: (x1, y1, x2, y2)

        Returns:
            List of (x, y) pixel coordinates
        """

        if landmarks is None:
            return []

        x1, y1, x2, y2 = head_bbox

        roi_width = x2 - x1
        roi_height = y2 - y1

        points = []

        for landmark in landmarks.landmark:

            px = int(landmark.x * roi_width) + x1
            py = int(landmark.y * roi_height) + y1

            points.append((px, py))

        return points


    def draw_landmarks(self, frame, landmarks, head_bbox):
        """
        Draw ROI landmarks back on the original frame.
        """

        points= self.get_landmark_points(landmarks, head_bbox)
        for point in points:
            cv2.circle(frame, point, 1, (0,255,0),-1)