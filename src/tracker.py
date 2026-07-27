class Tracker:

    def process(self, result):

        tracks = []

        if result.boxes is None:
            return tracks

        if result.boxes.id is None:
            return tracks

        ids = result.boxes.id.int().cpu().tolist()
        boxes = result.boxes.xyxy.cpu().tolist()

        for track_id, box in zip(ids, boxes):
            tracks.append((track_id, box))

        return tracks