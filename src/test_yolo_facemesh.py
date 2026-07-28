import cv2
from camera import Camera
from detector import Detector
from face_mesh import FaceMeshDetector
import face_mesh

print("=" * 50)
print(face_mesh.__file__)
print(dir(FaceMeshDetector))
print("=" * 50)

from utils import extract_head_roi


camera = Camera()
detector = Detector()
face_detector = FaceMeshDetector(max_num_faces=1)

while True:

    ret, frame = camera.read()

    if not ret:
        break

    # YOLO Detection
    results = detector.detect(frame)

    annotated = results[0].plot()
    for box in results[0].boxes:

        # Bounding box
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

        bbox = (x1, y1, x2, y2)

        # Head ROI
        head_roi, head_bbox = extract_head_roi(frame, bbox)

        hx1, hy1, hx2, hy2 = head_bbox

        # Draw Head ROI
        cv2.rectangle(
            annotated,
            (hx1, hy1),
            (hx2, hy2),
            (0, 255, 0),
            2,
        )

        # Detect Face Mesh on ROI
        landmarks = face_detector.detect(head_roi)

        # Draw landmarks
        face_detector.draw_landmarks(
            annotated,
            landmarks,
            head_bbox,
        )
        
    cv2.imshow("YOLO + Face Mesh", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()