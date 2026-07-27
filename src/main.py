import cv2
import time

from camera import Camera
from detector import Detector
from tracker import Tracker
from visualizer import Visualizer
from student_manager import StudentManager


camera = Camera()

detector = Detector()

tracker = Tracker()

visualizer = Visualizer()

student_manager = StudentManager()

last_print_time = time.time()

while True:

    ret, frame = camera.read()

    if not ret:
        break

    results = detector.detect(frame)

    result = results[0]

    tracks = tracker.process(result)

    for track_id, box in tracks:
        student_manager.update_student(track_id, box)

    if time.time() - last_print_time >= 1:
        student_manager.print_summary()
        last_print_time = time.time()

    frame = visualizer.draw(result)

    cv2.imshow("Classroom Engagement System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()