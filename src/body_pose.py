import cv2
from ultralytics import YOLO


class BodyPoseEstimator:
    KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    ]
    def get_named_keypoints(self,person):
        keypoints = person["keypoints"]
        confidence = person["confidence"]
        named = {}
        for i, name in enumerate(self.KEYPOINT_NAMES):
            named[name] = {
                "point" : keypoints[i],
                "confidence": (
                    confidence[i]
                    if confidence is not None
                    else None
                ),
            }
        return named
    
    def __init__(self, model_path="yolo11n-pose.pt"):
        self.model = YOLO(model_path)

    def estimate(self, frame):
        results = self.model(
            frame,
            verbose=False
        )

        return results[0]

    def track(
        self,
        frame,
        conf=0.5,
        iou=0.5,
        tracker="bytetrack.yaml",
    ):
        """
        Detect, track and estimate the pose of every person in ONE model
        pass (replaces the separate person detector + IoU matching).

        Returns:
            list of (track_id, [x1, y1, x2, y2]), and
            dict {track_id: person} where person has "keypoints" and
            "confidence" like extract_keypoints() returns.
            Both are empty until the tracker has assigned IDs.
        """

        results = self.model.track(
            frame,
            persist=True,
            tracker=tracker,
            conf=conf,
            iou=iou,
            verbose=False,
        )

        result = results[0]

        if result.boxes is None or result.boxes.id is None:
            return [], {}

        ids = result.boxes.id.int().cpu().tolist()
        boxes = result.boxes.xyxy.cpu().tolist()

        # keypoints come in the same order as the boxes
        persons = self.extract_keypoints(result)

        tracks = list(zip(ids, boxes))

        by_id = {
            track_id: person
            for track_id, person in zip(ids, persons)
        }

        return tracks, by_id

    def reset_tracker(self):
        """Forget all tracks (call when the camera view changes)."""

        predictor = self.model.predictor

        if predictor is not None and hasattr(predictor, "trackers"):
            for tracker in predictor.trackers:
                tracker.reset()

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