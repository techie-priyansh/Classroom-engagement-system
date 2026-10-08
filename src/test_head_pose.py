import cv2
import numpy as np

from detector import Detector
from tracker import Tracker
from face_mesh import FaceMeshDetector
from utils import extract_head_roi


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

VIDEO_PATH = "tests/test1.mp4"
OUTPUT_PATH = "results/video_test/head_pose_axes.jpg"

TARGET_TRACK_ID = 2


# ---------------------------------------------------------
# Corrected 3D face model
# ---------------------------------------------------------

ORIGINAL_MODEL = np.array([
    (0.0, 0.0, 0.0),          # Nose
    (0.0, -63.6, -12.5),      # Chin
    (-43.3, 32.7, -26.0),     # Left eye
    (43.3, 32.7, -26.0),      # Right eye
    (-28.9, -28.9, -24.1),    # Left mouth
    (28.9, -28.9, -24.1)      # Right mouth
], dtype=np.float64)


# OpenCV camera-coordinate convention
MODEL_CV = ORIGINAL_MODEL * np.array(
    [1.0, -1.0, -1.0]
)


# MediaPipe landmark IDs
LANDMARK_IDS = [
    1,      # Nose
    152,    # Chin
    33,     # Left eye
    263,    # Right eye
    61,     # Left mouth
    291     # Right mouth
]


# ---------------------------------------------------------
# Make head ROI square
# ---------------------------------------------------------

def make_square_roi(frame, bbox):

    x1, y1, x2, y2 = map(int, bbox)

    width = x2 - x1
    height = y2 - y1

    side = max(width, height)

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    sx1 = cx - side // 2
    sy1 = cy - side // 2

    sx2 = sx1 + side
    sy2 = sy1 + side

    # Padding required if square goes outside frame
    pad_left = max(0, -sx1)
    pad_top = max(0, -sy1)
    pad_right = max(0, sx2 - frame.shape[1])
    pad_bottom = max(0, sy2 - frame.shape[0])

    padded = cv2.copyMakeBorder(
        frame,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_REPLICATE
    )

    # Shift coordinates into padded image
    psx1 = sx1 + pad_left
    psy1 = sy1 + pad_top
    psx2 = sx2 + pad_left
    psy2 = sy2 + pad_top

    roi = padded[
        psy1:psy2,
        psx1:psx2
    ]

    # Return bbox in ORIGINAL frame coordinates
    square_bbox = (
        sx1,
        sy1,
        sx2,
        sy2
    )

    return roi, square_bbox


# ---------------------------------------------------------
# Euler angle conversion
# ---------------------------------------------------------

def rotation_matrix_to_euler_angles(R):

    sy = np.sqrt(
        R[0, 0] ** 2 +
        R[1, 0] ** 2
    )

    singular = sy < 1e-6

    if not singular:

        pitch = np.arctan2(
            R[2, 1],
            R[2, 2]
        )

        yaw = np.arctan2(
            -R[2, 0],
            sy
        )

        roll = np.arctan2(
            R[1, 0],
            R[0, 0]
        )

    else:

        pitch = np.arctan2(
            -R[1, 2],
            R[1, 1]
        )

        yaw = np.arctan2(
            -R[2, 0],
            sy
        )

        roll = 0

    return (
        np.degrees(yaw),
        np.degrees(pitch),
        np.degrees(roll)
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    detector = Detector()
    tracker = Tracker()
    face_mesh = FaceMeshDetector()

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    found = False

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # -------------------------------------------------
        # YOLO detection
        # -------------------------------------------------

        results = detector.detect(frame)

        if len(results) == 0:
            continue

        result = results[0]

        # -------------------------------------------------
        # Tracking
        # -------------------------------------------------

        tracks = tracker.process(result)

        for track_id, bbox in tracks:

            if track_id != TARGET_TRACK_ID:
                continue

            print(f"Processing Track ID: {track_id}")

            # -------------------------------------------------
            # Original head ROI
            # -------------------------------------------------

            head_roi, head_bbox = extract_head_roi(
                frame,
                bbox
            )

            # -------------------------------------------------
            # Convert head ROI into square ROI
            # -------------------------------------------------

            square_roi, square_bbox = make_square_roi(
                frame,
                head_bbox
            )

            # -------------------------------------------------
            # Face Mesh
            # -------------------------------------------------

            landmarks = face_mesh.detect(
                square_roi
            )

            if landmarks is None:
                print("No face detected.")
                continue

            landmark_points = face_mesh.get_landmark_points(
                landmarks,
                square_bbox
            )

            if landmark_points is None:
                print("No landmark points.")
                continue

            # -------------------------------------------------
            # Select six landmarks
            # -------------------------------------------------

            image_points = np.array([
                landmark_points[1],
                landmark_points[152],
                landmark_points[33],
                landmark_points[263],
                landmark_points[61],
                landmark_points[291]
            ], dtype=np.float64)

            # -------------------------------------------------
            # Camera matrix
            # -------------------------------------------------

            h, w = frame.shape[:2]

            focal_length = max(w, h)

            center = (
                w / 2,
                h / 2
            )

            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float64)

            dist_coeffs = np.zeros(
                (5, 1),
                dtype=np.float64
            )

            # -------------------------------------------------
            # Solve PnP
            # -------------------------------------------------

            success, rotation_vector, translation_vector = cv2.solvePnP(
                MODEL_CV,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success:
                print("solvePnP failed.")
                continue

            # -------------------------------------------------
            # Rotation matrix
            # -------------------------------------------------

            rotation_matrix, _ = cv2.Rodrigues(
                rotation_vector
            )

            yaw, pitch, roll = rotation_matrix_to_euler_angles(
                rotation_matrix
            )

            # -------------------------------------------------
            # Print result
            # -------------------------------------------------

            print()
            print("HEAD POSE (corrected 3D model)")
            print("--------------------------------")
            print(f"Track ID: {track_id}")
            print(f"Yaw   : {yaw:.2f}°")
            print(f"Pitch : {pitch:.2f}°")
            print(f"Roll  : {roll:.2f}°")

            print()
            print(
                "Translation (mm):",
                translation_vector.flatten()
            )

            print(
                f"tz = {translation_vector[2][0]:.2f}"
            )

            # -------------------------------------------------
            # Draw face landmarks
            # -------------------------------------------------

            for point_id in LANDMARK_IDS:

                x, y = landmark_points[point_id]

                cv2.circle(
                    frame,
                    (int(x), int(y)),
                    5,
                    (0, 255, 255),
                    -1
                )

            # -------------------------------------------------
            # Draw coordinate axes
            # -------------------------------------------------

            cv2.drawFrameAxes(
                frame,
                camera_matrix,
                dist_coeffs,
                rotation_vector,
                translation_vector,
                100,
                4
            )

            # -------------------------------------------------
            # Draw student bounding box
            # -------------------------------------------------

            x1, y1, x2, y2 = map(
                int,
                bbox
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"ID: {track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # -------------------------------------------------
            # Display pose
            # -------------------------------------------------

            info_y = max(30, y1 - 80)

            cv2.putText(
                frame,
                f"Yaw: {yaw:.1f}",
                (x1, info_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Pitch: {pitch:.1f}",
                (x1, info_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Roll: {roll:.1f}",
                (x1, info_y + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2
            )

            # -------------------------------------------------
            # Save result
            # -------------------------------------------------

            cv2.imwrite(
                OUTPUT_PATH,
                frame
            )

            print()
            print(f"Saved: {OUTPUT_PATH}")
            print(
                "Axes: X=red, Y=green, Z=blue"
            )

            found = True
            break

        if found:
            break

    cap.release()

    if not found:
        print(
            f"Track ID {TARGET_TRACK_ID} "
            "was not found."
        )


if __name__ == "__main__":
    main()