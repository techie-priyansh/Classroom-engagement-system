from ultralytics import YOLO


class Detector:
    def __init__(
        self,
        model_name="yolo11s.pt",
        tracker="bytetrack.yaml",
        conf=0.5,
        iou=0.5,
    ):
        self.model = YOLO(model_name)

        self.tracker = tracker
        self.conf = conf
        self.iou = iou

    def detect(self, frame):

        return self.model.track(
            frame,
            persist=True,
            tracker=self.tracker,
            classes=[0],
            conf=self.conf,
            iou=self.iou,
            verbose=False,
        )