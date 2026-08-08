import cv2
from ultralytics import YOLO


class BodyPoseEstimator:

    def __init__(self, model_path="yolo11n-pose.pt"):
        self.model = YOLO(model_path)

    def estimate(self, frame):
        results = self.model(
            frame,
            verbose=False
        )

        return results[0]

    def extract_keypoints(self, results):
        """
        Extract pose keypoints for every detected person.

        Returns:
            List of dictionaries, one per detected person.
        """

        if results is None or results.keypoints is None:
            return []

        keypoints = results.keypoints

        xy = keypoints.xy.cpu().numpy()
        confidence = keypoints.conf

        if confidence is not None:
            confidence = confidence.cpu().numpy()

        persons = []

        for person_idx in range(len(xy)):

            person = {
                "keypoints": xy[person_idx],
                "confidence": (
                    confidence[person_idx]
                    if confidence is not None
                    else None
                ),
            }

            persons.append(person)

        return persons

    def draw(self, frame, results):

        if results is None:
            return frame

        return results.plot()