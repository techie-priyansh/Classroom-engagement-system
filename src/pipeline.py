import cv2

from detector import Detector
from tracker import Tracker
from student_manager import StudentManager
from face_mesh import FaceMeshDetector
from head_pose import HeadPoseEstimator
from eye_analyser import EyeAnalyser
from utils import extract_head_roi
from gaze_estimator import GazeEstimator

#file_path = "/home/mylinux/Desktop/MTech_Project/MTP/Classroom-engagement-system/215475_medium.mp4"
def main():

    # --------------------------------------------------
    # Initialize Modules
    # --------------------------------------------------
    detector = Detector()
    tracker = Tracker()
    student_manager = StudentManager()

    face_mesh = FaceMeshDetector()
    head_pose = HeadPoseEstimator()
    eye_analyzer = EyeAnalyser()
    gaze_estimator= GazeEstimator()

    #cap = cv2.VideoCapture(file_path)

    cap = cv2.VideoCapture(2)

    if not cap.isOpened():
        print("Error: Unable to open camera.")
        return

    # --------------------------------------------------
    # Main Loop
    # --------------------------------------------------
    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # ----------------------------------------------
        # Detection + Tracking
        # ----------------------------------------------
        results = detector.detect(frame)

        if len(results) == 0:
            cv2.imshow("Behavior Intelligence Pipeline", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            continue

        result = results[0]

        tracks = tracker.process(result)

        # ----------------------------------------------
        # Process every detected student
        # ----------------------------------------------
        for track_id, bbox in tracks:

            student_manager.update_student(track_id, bbox)

            student = student_manager.get_students()[track_id]

            x1, y1, x2, y2 = map(int, bbox)

            # ------------------------------------------
            # Draw Bounding Box
            # ------------------------------------------
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"ID: {track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            # ------------------------------------------
            # Head ROI
            # ------------------------------------------
            head_roi, head_bbox = extract_head_roi(frame, bbox)

            # ------------------------------------------
            # Face Mesh
            # ------------------------------------------
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

            student.landmark_points = landmark_points

            # ------------------------------------------
            # Head Pose
            # ------------------------------------------
            pose = head_pose.estimate(
                landmark_points,
                frame.shape,
            )

            # ------------------------------------------
# Head Pose + Eye Analysis + Gaze
# ------------------------------------------

        if pose is not None:

            student.head_pose = pose

            yaw, pitch, roll = pose

            eye_state, left_ear, right_ear = eye_analyzer.analyse(
                landmark_points
            )

            gaze_direction = gaze_estimator.estimate(
                landmark_points,
                pose,
            )

            if eye_state is not None:

                avg_ear = (left_ear + right_ear) / 2

                student.eye_state = eye_state
                student.left_ear = left_ear
                student.right_ear = right_ear
                student.avg_ear = avg_ear
                student.gaze_direction = gaze_direction

                info_y = max(20, y1 - 100)

                cv2.putText(
                    frame,
                    f"ID: {track_id}",
                    (x1, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0,255,0),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Yaw : {yaw:.1f}",
                    (x1, info_y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0,255,255),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Pitch : {pitch:.1f}",
                    (x1, info_y + 36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0,255,255),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Roll : {roll:.1f}",
                    (x1, info_y + 54),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0,255,255),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Eyes : {eye_state}",
                    (x1, info_y + 72),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255,255,0),
                    2,
                )

                cv2.putText(
                    frame,
                    f"EAR : {avg_ear:.3f}",
                    (x1, info_y + 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255,255,0),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Gaze : {gaze_direction}",
                    (x1, info_y + 108),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0,255,255),
                    2,
                )
        # ----------------------------------------------
        # Display
        # ----------------------------------------------
        cv2.imshow(
            "Behavior Intelligence Pipeline",
            frame,
        )

        if cv2.waitKey(1) & 0xFF == ord("q"): 
            break

    # --------------------------------------------------
    # Cleanup
    # --------------------------------------------------
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()