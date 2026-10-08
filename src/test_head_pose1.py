# import cv2
# import numpy as np

# from detector import Detector
# from tracker import Tracker
# from face_mesh import FaceMeshDetector
# from utils import extract_head_roi


# VIDEO_PATH = "tests/test1.mp4"


# def rotation_to_euler(R):
#     sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)

#     singular = sy < 1e-6

#     if not singular:
#         pitch = np.arctan2(R[2, 1], R[2, 2])
#         yaw = np.arctan2(-R[2, 0], sy)
#         roll = np.arctan2(R[1, 0], R[0, 0])
#     else:
#         pitch = np.arctan2(-R[1, 2], R[1, 1])
#         yaw = np.arctan2(-R[2, 0], sy)
#         roll = 0

#     return np.degrees([
#         yaw,
#         pitch,
#         roll
#     ])


# def run_pnp(model_points, image_points, camera_matrix, dist_coeffs):

#     success, rvec, tvec = cv2.solvePnP(
#         model_points,
#         image_points,
#         camera_matrix,
#         dist_coeffs,
#         flags=cv2.SOLVEPNP_ITERATIVE
#     )

#     if not success:
#         return None

#     R, _ = cv2.Rodrigues(rvec)

#     return rotation_to_euler(R)


# def main():

#     cap = cv2.VideoCapture(VIDEO_PATH)

#     if not cap.isOpened():
#         print("Could not open video")
#         return

#     detector = Detector()
#     tracker = Tracker()
#     face_mesh = FaceMeshDetector()

#     # --------------------------------------------------
#     # Find first usable face
#     # --------------------------------------------------

#     while True:

#         ret, frame = cap.read()

#         if not ret:
#             print("No suitable face found")
#             break

#         results = detector.detect(frame)

#         if len(results) == 0:
#             continue

#         tracks = tracker.process(results[0])

#         if len(tracks) == 0:
#             continue

#         for track_id, bbox in tracks:

#             head_roi, head_bbox = extract_head_roi(
#                 frame,
#                 bbox
#             )

#             landmarks = face_mesh.detect(head_roi)

#             if landmarks is None:
#                 continue

#             landmark_points = face_mesh.get_landmark_points(
#                 landmarks,
#                 head_bbox
#             )

#             # --------------------------------------------------
#             # 2D image points
#             # --------------------------------------------------

#             image_points = np.array([
#                 landmark_points[1],    # nose
#                 landmark_points[152],  # chin
#                 landmark_points[33],   # left eye
#                 landmark_points[263],  # right eye
#                 landmark_points[61],   # left mouth
#                 landmark_points[291]   # right mouth
#             ], dtype=np.float64)

#             # --------------------------------------------------
#             # Original 3D model
#             # --------------------------------------------------

#             original_model = np.array([
#                 (0.0, 0.0, 0.0),
#                 (0.0, -63.6, -12.5),
#                 (-43.3, 32.7, -26.0),
#                 (43.3, 32.7, -26.0),
#                 (-28.9, -28.9, -24.1),
#                 (28.9, -28.9, -24.1)
#             ], dtype=np.float64)

#             # --------------------------------------------------
#             # Completely flipped coordinate system
#             # --------------------------------------------------

#             flipped_model = -original_model

#             # --------------------------------------------------
#             # Camera matrix
#             # --------------------------------------------------

#             h, w = frame.shape[:2]

#             focal_length = max(w, h)

#             camera_matrix = np.array([
#                 [focal_length, 0, w / 2],
#                 [0, focal_length, h / 2],
#                 [0, 0, 1]
#             ], dtype=np.float64)

#             dist_coeffs = np.zeros(
#                 (5, 1),
#                 dtype=np.float64
#             )

#             # --------------------------------------------------
#             # Run both models
#             # --------------------------------------------------

#             original_angles = run_pnp(
#                 original_model,
#                 image_points,
#                 camera_matrix,
#                 dist_coeffs
#             )

#             flipped_angles = run_pnp(
#                 flipped_model,
#                 image_points,
#                 camera_matrix,
#                 dist_coeffs
#             )

#             # --------------------------------------------------
#             # Results
#             # --------------------------------------------------

#             print()
#             print("=" * 70)
#             print("3D MODEL COORDINATE TEST")
#             print("=" * 70)

#             print(f"Track ID: {track_id}")

#             print("\nOriginal model:")
#             print(
#                 f"Yaw   : {original_angles[0]:.2f}°"
#             )
#             print(
#                 f"Pitch : {original_angles[1]:.2f}°"
#             )
#             print(
#                 f"Roll  : {original_angles[2]:.2f}°"
#             )

#             print("\nFlipped model (-X, -Y, -Z):")
#             print(
#                 f"Yaw   : {flipped_angles[0]:.2f}°"
#             )
#             print(
#                 f"Pitch : {flipped_angles[1]:.2f}°"
#             )
#             print(
#                 f"Roll  : {flipped_angles[2]:.2f}°"
#             )

#             print("=" * 70)

#             cap.release()
#             return

#     cap.release()


# if __name__ == "__main__":
#     main()


################
###Claude code##

import os

import cv2
import numpy as np

from detector import Detector
from tracker import Tracker
from face_mesh import FaceMeshDetector
from utils import extract_head_roi


VIDEO_PATH = "tests/test1.mp4"
OUTPUT_DIR = "results/video_test"

# MediaPipe warns "NORM_RECT without IMAGE_DIMENSIONS ... square ROI" when the
# crop is not square. Set True to pad the head crop to a square before
# running FaceMesh. Assumes head_bbox is (x1, y1, x2, y2) in frame pixels.
USE_SQUARE_ROI = True


# ----------------------------------------------------------------------
# 3D face model (generic head, millimetres)
# Original convention: X right, Y UP, Z toward the viewer (nose sticks out).
# OpenCV camera convention: X right, Y DOWN, Z away from camera.
# Going from one to the other is a 180 degree rotation about X, which flips
# Y and Z only. (-model flips X as well -> reflection, not a valid rotation.)
# ----------------------------------------------------------------------
ORIGINAL_MODEL = np.array([
    (0.0, 0.0, 0.0),         # nose tip        (landmark 1)
    (0.0, -63.6, -12.5),     # chin            (landmark 152)
    (-43.3, 32.7, -26.0),    # left eye corner (landmark 33)
    (43.3, 32.7, -26.0),     # right eye corner(landmark 263)
    (-28.9, -28.9, -24.1),   # left mouth      (landmark 61)
    (28.9, -28.9, -24.1),    # right mouth     (landmark 291)
], dtype=np.float64)

MODEL_CV = ORIGINAL_MODEL * np.array([1.0, -1.0, -1.0])

LANDMARK_IDS = [1, 152, 33, 263, 61, 291]


def rotation_to_euler(R):
    """Rotation matrix -> (yaw, pitch, roll) in degrees."""
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        pitch = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(-R[2, 0], sy)
        roll = np.arctan2(R[1, 0], R[0, 0])
    else:
        pitch = np.arctan2(-R[1, 2], R[1, 1])
        yaw = np.arctan2(-R[2, 0], sy)
        roll = 0.0

    return np.degrees([yaw, pitch, roll])


def run_pnp(model_points, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE):
    """Returns (euler_angles, rvec, tvec) or None."""
    success, rvec, tvec = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=flags
    )

    if not success:
        return None

    R, _ = cv2.Rodrigues(rvec)
    return rotation_to_euler(R), rvec, tvec


def make_square_roi(frame, head_bbox):
    """Pad the head crop to a square (replicating border pixels where the
    square falls outside the frame). Returns (square_roi, square_bbox) where
    square_bbox is (x1, y1, x2, y2) in frame coordinates (may be negative or
    beyond frame size)."""
    x1, y1, x2, y2 = [int(v) for v in head_bbox]
    w, h = x2 - x1, y2 - y1
    side = max(w, h)

    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    sx1, sy1 = cx - side // 2, cy - side // 2
    sx2, sy2 = sx1 + side, sy1 + side

    fh, fw = frame.shape[:2]
    pad_l = max(0, -sx1)
    pad_t = max(0, -sy1)
    pad_r = max(0, sx2 - fw)
    pad_b = max(0, sy2 - fh)

    cx1, cy1 = max(0, sx1), max(0, sy1)
    cx2, cy2 = min(fw, sx2), min(fh, sy2)
    crop = frame[cy1:cy2, cx1:cx2]

    if pad_l or pad_t or pad_r or pad_b:
        crop = cv2.copyMakeBorder(
            crop, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REPLICATE
        )

    return crop, (sx1, sy1, sx2, sy2)


def main():

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("Could not open video")
        return

    detector = Detector()
    tracker = Tracker()
    face_mesh = FaceMeshDetector()

    # --------------------------------------------------
    # Find first usable face
    # --------------------------------------------------

    while True:

        ret, frame = cap.read()

        if not ret:
            print("No suitable face found")
            break

        results = detector.detect(frame)

        if len(results) == 0:
            continue

        tracks = tracker.process(results[0])

        if len(tracks) == 0:
            continue

        for track_id, bbox in tracks:

            head_roi, head_bbox = extract_head_roi(frame, bbox)

            if USE_SQUARE_ROI:
                head_roi, head_bbox = make_square_roi(frame, head_bbox)

            if head_roi is None or head_roi.size == 0:
                continue

            landmarks = face_mesh.detect(head_roi)

            if landmarks is None:
                continue

            landmark_points = face_mesh.get_landmark_points(
                landmarks,
                head_bbox
            )

            # --------------------------------------------------
            # 2D image points (same order as MODEL_CV)
            # --------------------------------------------------

            image_points = np.array(
                [landmark_points[i] for i in LANDMARK_IDS],
                dtype=np.float64
            )

            # --------------------------------------------------
            # Camera matrix
            # --------------------------------------------------

            h, w = frame.shape[:2]
            focal_length = max(w, h)

            camera_matrix = np.array([
                [focal_length, 0, w / 2],
                [0, focal_length, h / 2],
                [0, 0, 1]
            ], dtype=np.float64)

            dist_coeffs = np.zeros((5, 1), dtype=np.float64)

            # --------------------------------------------------
            # Run PnP: old (Y-up) model vs corrected OpenCV model
            # --------------------------------------------------

            old = run_pnp(ORIGINAL_MODEL, image_points,
                          camera_matrix, dist_coeffs)
            new = run_pnp(MODEL_CV, image_points,
                          camera_matrix, dist_coeffs)

            if old is None or new is None:
                continue

            old_angles, _, _ = old
            new_angles, rvec, tvec = new

            print()
            print("=" * 70)
            print("HEAD POSE (corrected 3D model)")
            print("=" * 70)
            print(f"Track ID: {track_id}")

            print("\nOld model (Y-up, Z-toward-camera):")
            print(f"Yaw   : {old_angles[0]:.2f}°")
            print(f"Pitch : {old_angles[1]:.2f}°")
            print(f"Roll  : {old_angles[2]:.2f}°")

            print("\nCorrected model (Y-down, Z-away, OpenCV convention):")
            print(f"Yaw   : {new_angles[0]:.2f}°")
            print(f"Pitch : {new_angles[1]:.2f}°")
            print(f"Roll  : {new_angles[2]:.2f}°")

            print(f"\nTranslation (mm): {tvec.ravel().round(1)}")
            print("(tz should be positive: face is in front of camera)")
            print("=" * 70)

            # --------------------------------------------------
            # Visual sanity check: draw axes on the face
            # --------------------------------------------------

            vis = frame.copy()
            for (px, py) in image_points:
                cv2.circle(vis, (int(round(px)), int(round(py))),
                           2, (0, 255, 0), -1)

            cv2.drawFrameAxes(
                vis, camera_matrix, dist_coeffs, rvec, tvec, 40
            )

            os.makedirs(OUTPUT_DIR, exist_ok=True)
            out_path = os.path.join(OUTPUT_DIR, "head_pose_axes_main.jpg")
            cv2.imwrite(out_path, vis)
            print(f"Saved: {out_path}")
            print("Axes: X=red (right), Y=green (down), Z=blue (into the face)")

            cap.release()
            return

    cap.release()


if __name__ == "__main__":
    main()


# import os

# import cv2
# import numpy as np

# from detector import Detector
# from tracker import Tracker
# from face_mesh import FaceMeshDetector
# from utils import extract_head_roi


# VIDEO_PATH = "tests/test1.mp4"
# OUTPUT_DIR = "results/video_test"

# # MediaPipe warns "NORM_RECT without IMAGE_DIMENSIONS ... square ROI" when the
# # crop is not square. Set True to pad the head crop to a square before
# # running FaceMesh. Assumes head_bbox is (x1, y1, x2, y2) in frame pixels.
# USE_SQUARE_ROI = True


# # ----------------------------------------------------------------------
# # 3D face model (generic head, millimetres)
# # Original convention: X right, Y UP, Z toward the viewer (nose sticks out).
# # OpenCV camera convention: X right, Y DOWN, Z away from camera.
# # Going from one to the other is a 180 degree rotation about X, which flips
# # Y and Z only. (-model flips X as well -> reflection, not a valid rotation.)
# # ----------------------------------------------------------------------
# ORIGINAL_MODEL = np.array([
#     (0.0, 0.0, 0.0),         # nose tip        (landmark 1)
#     (0.0, -63.6, -12.5),     # chin            (landmark 152)
#     (-43.3, 32.7, -26.0),    # left eye corner (landmark 33)
#     (43.3, 32.7, -26.0),     # right eye corner(landmark 263)
#     (-28.9, -28.9, -24.1),   # left mouth      (landmark 61)
#     (28.9, -28.9, -24.1),    # right mouth     (landmark 291)
# ], dtype=np.float64)

# MODEL_CV = ORIGINAL_MODEL * np.array([1.0, -1.0, -1.0])

# LANDMARK_IDS = [1, 152, 33, 263, 61, 291]


# def rotation_to_euler(R):
#     """Rotation matrix -> (yaw, pitch, roll) in degrees."""
#     sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
#     singular = sy < 1e-6

#     if not singular:
#         pitch = np.arctan2(R[2, 1], R[2, 2])
#         yaw = np.arctan2(-R[2, 0], sy)
#         roll = np.arctan2(R[1, 0], R[0, 0])
#     else:
#         pitch = np.arctan2(-R[1, 2], R[1, 1])
#         yaw = np.arctan2(-R[2, 0], sy)
#         roll = 0.0

#     return np.degrees([yaw, pitch, roll])


# def run_pnp(model_points, image_points, camera_matrix, dist_coeffs,
#             flags=cv2.SOLVEPNP_ITERATIVE):
#     """Returns (euler_angles, rvec, tvec) or None."""
#     success, rvec, tvec = cv2.solvePnP(
#         model_points,
#         image_points,
#         camera_matrix,
#         dist_coeffs,
#         flags=flags
#     )

#     if not success:
#         return None

#     R, _ = cv2.Rodrigues(rvec)
#     return rotation_to_euler(R), rvec, tvec


# def make_square_roi(frame, head_bbox):
#     """Pad the head crop to a square (replicating border pixels where the
#     square falls outside the frame). Returns (square_roi, square_bbox) where
#     square_bbox is (x1, y1, x2, y2) in frame coordinates (may be negative or
#     beyond frame size)."""
#     x1, y1, x2, y2 = [int(v) for v in head_bbox]
#     w, h = x2 - x1, y2 - y1
#     side = max(w, h)

#     cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
#     sx1, sy1 = cx - side // 2, cy - side // 2
#     sx2, sy2 = sx1 + side, sy1 + side

#     fh, fw = frame.shape[:2]
#     pad_l = max(0, -sx1)
#     pad_t = max(0, -sy1)
#     pad_r = max(0, sx2 - fw)
#     pad_b = max(0, sy2 - fh)

#     cx1, cy1 = max(0, sx1), max(0, sy1)
#     cx2, cy2 = min(fw, sx2), min(fh, sy2)
#     crop = frame[cy1:cy2, cx1:cx2]

#     if pad_l or pad_t or pad_r or pad_b:
#         crop = cv2.copyMakeBorder(
#             crop, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REPLICATE
#         )

#     return crop, (sx1, sy1, sx2, sy2)


# def draw_pose_axes(frame, image_points, rvec, tvec,
#                    camera_matrix, dist_coeffs, out_size=700, axis_len=80.0):
#     """Draw the head axes on a zoomed-in, upscaled crop around the face so
#     they are clearly visible even when the face is only ~30 px wide in the
#     original frame.

#     X = red (subject's right in image), Y = green (down), Z = blue
#     (pointing OUT of the nose toward the camera).
#     """
#     # Axes in the MODEL_CV frame (origin = nose tip). Z is drawn toward the
#     # camera (negative Z in the OpenCV-style model), like a pointer from the nose.
#     axes_3d = np.float64([
#         [0, 0, 0],
#         [axis_len, 0, 0],
#         [0, axis_len, 0],
#         [0, 0, -axis_len],
#     ])

#     proj, _ = cv2.projectPoints(
#         axes_3d, rvec, tvec, camera_matrix, dist_coeffs
#     )
#     proj = proj.reshape(-1, 2)

#     # Crop box that contains all landmarks AND the axis tips, with margin
#     all_pts = np.vstack([image_points, proj])
#     x_min, y_min = all_pts.min(axis=0)
#     x_max, y_max = all_pts.max(axis=0)
#     margin = 0.35 * max(x_max - x_min, y_max - y_min)

#     box = (
#         int(x_min - margin), int(y_min - margin),
#         int(x_max + margin), int(y_max + margin)
#     )

#     crop, (sx1, sy1, sx2, sy2) = make_square_roi(frame, box)
#     side = crop.shape[0]
#     scale = out_size / float(side)

#     vis = cv2.resize(
#         crop, (out_size, out_size), interpolation=cv2.INTER_CUBIC
#     )

#     def to_vis(p):
#         return (
#             int(round((p[0] - sx1) * scale)),
#             int(round((p[1] - sy1) * scale))
#         )

#     # Landmarks
#     for p in image_points:
#         cv2.circle(vis, to_vis(p), 5, (0, 255, 255), -1, cv2.LINE_AA)
#         cv2.circle(vis, to_vis(p), 5, (0, 0, 0), 1, cv2.LINE_AA)

#     # Axes (BGR colours) with black outline so they stand out
#     origin = to_vis(proj[0])
#     axes_info = [
#         (proj[1], (0, 0, 255), "X"),
#         (proj[2], (0, 255, 0), "Y"),
#         (proj[3], (255, 0, 0), "Z"),
#     ]
#     for tip, color, label in axes_info:
#         end = to_vis(tip)
#         cv2.arrowedLine(vis, origin, end, (0, 0, 0), 8,
#                         cv2.LINE_AA, tipLength=0.15)
#         cv2.arrowedLine(vis, origin, end, color, 4,
#                         cv2.LINE_AA, tipLength=0.15)
#         cv2.putText(vis, label, (end[0] + 8, end[1] - 8),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
#         cv2.putText(vis, label, (end[0] + 8, end[1] - 8),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

#     return vis


# def main():

#     cap = cv2.VideoCapture(VIDEO_PATH)

#     if not cap.isOpened():
#         print("Could not open video")
#         return

#     detector = Detector()
#     tracker = Tracker()
#     face_mesh = FaceMeshDetector()

#     # --------------------------------------------------
#     # Find first usable face
#     # --------------------------------------------------

#     while True:

#         ret, frame = cap.read()

#         if not ret:
#             print("No suitable face found")
#             break

#         results = detector.detect(frame)

#         if len(results) == 0:
#             continue

#         tracks = tracker.process(results[0])

#         if len(tracks) == 0:
#             continue

#         for track_id, bbox in tracks:

#             head_roi, head_bbox = extract_head_roi(frame, bbox)

#             if USE_SQUARE_ROI:
#                 head_roi, head_bbox = make_square_roi(frame, head_bbox)

#             if head_roi is None or head_roi.size == 0:
#                 continue

#             landmarks = face_mesh.detect(head_roi)

#             if landmarks is None:
#                 continue

#             landmark_points = face_mesh.get_landmark_points(
#                 landmarks,
#                 head_bbox
#             )

#             # --------------------------------------------------
#             # 2D image points (same order as MODEL_CV)
#             # --------------------------------------------------

#             image_points = np.array(
#                 [landmark_points[i] for i in LANDMARK_IDS],
#                 dtype=np.float64
#             )

#             # --------------------------------------------------
#             # Camera matrix
#             # --------------------------------------------------

#             h, w = frame.shape[:2]
#             focal_length = max(w, h)

#             camera_matrix = np.array([
#                 [focal_length, 0, w / 2],
#                 [0, focal_length, h / 2],
#                 [0, 0, 1]
#             ], dtype=np.float64)

#             dist_coeffs = np.zeros((5, 1), dtype=np.float64)

#             # --------------------------------------------------
#             # Run PnP: old (Y-up) model vs corrected OpenCV model
#             # --------------------------------------------------

#             old = run_pnp(ORIGINAL_MODEL, image_points,
#                           camera_matrix, dist_coeffs)
#             new = run_pnp(MODEL_CV, image_points,
#                           camera_matrix, dist_coeffs)

#             if old is None or new is None:
#                 continue

#             old_angles, _, _ = old
#             new_angles, rvec, tvec = new

#             print()
#             print("=" * 70)
#             print("HEAD POSE (corrected 3D model)")
#             print("=" * 70)
#             print(f"Track ID: {track_id}")

#             print("\nOld model (Y-up, Z-toward-camera):")
#             print(f"Yaw   : {old_angles[0]:.2f}°")
#             print(f"Pitch : {old_angles[1]:.2f}°")
#             print(f"Roll  : {old_angles[2]:.2f}°")

#             print("\nCorrected model (Y-down, Z-away, OpenCV convention):")
#             print(f"Yaw   : {new_angles[0]:.2f}°")
#             print(f"Pitch : {new_angles[1]:.2f}°")
#             print(f"Roll  : {new_angles[2]:.2f}°")

#             print(f"\nTranslation (mm): {tvec.ravel().round(1)}")
#             print("(tz should be positive: face is in front of camera)")
#             print("=" * 70)

#             # --------------------------------------------------
#             # Visual sanity check: draw axes on the face
#             # --------------------------------------------------

#             vis = draw_pose_axes(
#                 frame, image_points, rvec, tvec,
#                 camera_matrix, dist_coeffs
#             )

#             os.makedirs(OUTPUT_DIR, exist_ok=True)
#             out_path = os.path.join(OUTPUT_DIR, "head_pose_axes.jpg")
#             cv2.imwrite(out_path, vis)
#             print(f"Saved: {out_path}")
#             print("Axes: X=red (right), Y=green (down), "
#                   "Z=blue (out of the nose, toward camera)")

#             cap.release()
#             return

#     cap.release()


# if __name__ == "__main__":
#     main()