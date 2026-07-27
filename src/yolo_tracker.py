import cv2
from ultralytics import YOLO
from student_manager import StudentManager
import time
# Load YOLO model
#model = YOLO("yolo11n.pt")

#Using s model than n
model = YOLO("yolo11s.pt")
student_manager= StudentManager()

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit.")
last_print_time = time.time()
while True:
    ret, frame = cap.read()

    if not ret:
        break

    # YOLO + ByteTrack
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],      # Only detect persons
        conf= 0.5,
        iou= 0.5,
        verbose=False
    )
    result= results[0]
    if result.boxes is not None and result.boxes.id is not None:
        track_ids = result.boxes.id.int().cpu().tolist()
        boxes= result.boxes.xyxy.cpu().tolist()
        for track_id, box in zip(track_ids, boxes):
            student_manager.update_student(track_id, box)

    annotated_frame = results[0].plot()
    if time.time() - last_print_time >= 1:
        student_manager.print_summary()
        last_print_time = time.time()
    cv2.imshow("YOLO11 + ByteTrack", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()