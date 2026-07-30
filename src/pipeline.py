import cv2

from detector import Detector
from tracker import Tracker
from student_manager import StudentManager
from face_mesh import FaceMeshDetector
from head_pose import HeadPoseEstimator
from utils import extract_head_roi


def main():

    detector = Detector()
    tracker = Tracker()
    student_manager = StudentManager()

    face_mesh = FaceMeshDetector()
    head_pose = HeadPoseEstimator()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to open camera.")
        return

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # ----------------------------
        # Student Detection + Tracking
        # ----------------------------
        results = detector.detect(frame)

        if len(results) > 0:

            result = results[0]

            tracks = tracker.process(result)

            for track_id, bbox in tracks:

                student_manager.update_student(track_id, bbox)

                x1, y1, x2, y2 = map(int, bbox)

                # Bounding Box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    f"ID {track_id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

                # ----------------------------
                # Head ROI
                # ----------------------------
                head_roi, head_bbox = extract_head_roi(
                    frame,
                    bbox,
                )

                # ----------------------------
                # Face Mesh
                # ----------------------------
                landmarks = face_mesh.detect(head_roi)

                if landmarks is None:
                    continue

                landmark_points = face_mesh.get_landmark_points(
                    landmarks,
                    head_bbox,
                )

                face_mesh.draw_landmarks(
                    frame,
                    landmark_points,
                )

                # ----------------------------
                # Head Pose
                # ----------------------------
                pose = head_pose.estimate(
                    landmark_points,
                    frame.shape,
                )

                if pose is None:
                    continue

                yaw, pitch, roll = pose
                student = student_manager.get_students()[track_id]
                student.landmark_points = landmark_points
                student.head_pose = pose
                cv2.putText(
                    frame,
                    f"Yaw : {yaw:.1f}",
                    (x1, y1 - 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Pitch : {pitch:.1f}",
                    (x1, y1 - 27),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Roll : {roll:.1f}",
                    (x1, y1 - 9),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2,
                )

        cv2.imshow("Behavior Intelligence Pipeline", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()