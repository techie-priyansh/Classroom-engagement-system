import csv
from pathlib import Path

import cv2
import numpy as np

from detector import Detector
from tracker import Tracker
from student_manager import StudentManager
from face_mesh import FaceMeshDetector
from head_pose import HeadPoseEstimator, AngleSmoother
from eye_analyser import EyeAnalyser
from gaze_estimator import GazeEstimator
from body_pose import BodyPoseEstimator
from posture_analyser import PostureAnalyser
from utils import extract_head_roi, make_square_roi
from settle import (
    find_settled_segments,
    segments_from_sweep_log,
    save_segments,
    SegmentLookup,
)


# --------------------------------------------------
# Paths
# --------------------------------------------------

VIDEO_PATH = Path("tests/NR131_test1_30Sec.mp4")
OUTPUT_DIR = Path("results/video_test")

OUTPUT_VIDEO = OUTPUT_DIR / "annotated1.mp4"
OUTPUT_CSV = OUTPUT_DIR / "student_data1.csv"


# --------------------------------------------------
# Settings
# --------------------------------------------------

SHOW_WINDOW = True          # set False on a headless machine
SMOOTH_HEAD_POSE = True     # EMA smoothing of yaw/pitch/roll per student
EMA_ALPHA = 0.4             # 1.0 = no smoothing, lower = smoother

# True : one pose model finds, tracks and gives keypoints (about 3x less
#        compute per frame in our CPU test; same people found).
# False: the older two-model path (person detector + pose model, matched
#        by IoU). Keep it for comparison.
USE_POSE_ONLY = True

MIN_POSE_IOU = 0.3          # minimum IoU to accept a body-pose match
KEYPOINT_CONF = 0.3         # keypoints below this confidence do not count for the pose box

# --------------------------------------------------
# PTZ settled segments
# --------------------------------------------------
# The frames where the camera stands still and in focus are found from
# image sharpness and motion (see settle.py). They are saved next to the
# results; delete the file if the video or the settle settings change.

SEGMENTS_FILE = OUTPUT_DIR / f"{VIDEO_PATH.stem}_segments.json"

# Optional: the sweep log written by ptz_record_sweep.py. When this file
# exists, its hold windows are used (and trimmed by the image checks).
SWEEP_LOG = VIDEO_PATH.with_suffix(".sweep.json")


# --------------------------------------------------
# Utility
# --------------------------------------------------

def calculate_iou(box1, box2):
    """
    IoU between two boxes [x1, y1, x2, y2].
    """

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union


def pose_bounding_box(person):
    """
    Bounding box of the VALID keypoints of one body-pose person.
    Undetected keypoints (0, 0) or low-confidence ones are ignored, so
    they cannot stretch the box to the image corner.
    """

    keypoints = np.asarray(person["keypoints"], dtype=np.float64)

    valid = (keypoints[:, 0] > 0) & (keypoints[:, 1] > 0)

    # Confidence comes as a separate array (see BodyPoseEstimator).
    confidence = person.get("confidence")

    if confidence is not None:
        valid &= np.asarray(confidence) > KEYPOINT_CONF
    elif keypoints.shape[1] >= 3:
        valid &= keypoints[:, 2] > KEYPOINT_CONF

    if valid.sum() < 3:
        return None

    points = keypoints[valid]

    return [
        points[:, 0].min(),
        points[:, 1].min(),
        points[:, 0].max(),
        points[:, 1].max(),
    ]


def match_poses(tracks, pose_persons):
    """
    One-to-one matching of tracked students to body-pose persons.
    Greedy by highest IoU, with a minimum IoU threshold.

    Returns:
        dict {track_id: person}
    """

    pose_boxes = [pose_bounding_box(p) for p in pose_persons]

    candidates = []

    for track_id, bbox in tracks:
        for p_idx, pose_box in enumerate(pose_boxes):

            if pose_box is None:
                continue

            iou = calculate_iou(bbox, pose_box)

            if iou >= MIN_POSE_IOU:
                candidates.append((iou, track_id, p_idx))

    candidates.sort(key=lambda c: c[0], reverse=True)

    matches = {}
    used_tracks = set()
    used_poses = set()

    for iou, track_id, p_idx in candidates:

        if track_id in used_tracks or p_idx in used_poses:
            continue

        matches[track_id] = pose_persons[p_idx]
        used_tracks.add(track_id)
        used_poses.add(p_idx)

    return matches


def fmt(value, digits):
    return None if value is None else round(value, digits)


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Initialize modules
    # --------------------------------------------------

    detector = None if USE_POSE_ONLY else Detector()
    tracker = Tracker()
    student_manager = StudentManager()

    face_mesh = FaceMeshDetector()
    head_pose = HeadPoseEstimator()
    angle_smoother = AngleSmoother(alpha=EMA_ALPHA)
    eye_analyzer = EyeAnalyser()
    gaze_estimator = GazeEstimator()

    body_pose = BodyPoseEstimator()
    posture_analyzer = PostureAnalyser()

    # --------------------------------------------------
    # Open video
    # --------------------------------------------------

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        print(f"Error: Unable to open video: {VIDEO_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print()
    print("=" * 60)
    print("VIDEO PIPELINE")
    print("=" * 60)
    print(f"Video        : {VIDEO_PATH}")
    print(f"Resolution   : {width} x {height}")
    print(f"FPS          : {fps}")
    print(f"Frames       : {total_frames}")
    print(f"Output video : {OUTPUT_VIDEO}")
    print(f"Output CSV   : {OUTPUT_CSV}")
    print("=" * 60)

    # --------------------------------------------------
    # Settled camera segments (replaces fixed frame ranges)
    # --------------------------------------------------

    if SWEEP_LOG.exists():
        segments = segments_from_sweep_log(SWEEP_LOG, VIDEO_PATH)
        save_segments(segments, SEGMENTS_FILE)
        print(f"Segments from sweep log: {SWEEP_LOG}")
    else:
        segments = find_settled_segments(
            VIDEO_PATH,
            cache_path=SEGMENTS_FILE,
        )
        print(f"Segments from image analysis (cache: {SEGMENTS_FILE})")

    if not segments:
        print("No settled camera segments found.")
        return

    for segment in segments:
        print(
            f"  Region {segment['region']}: frames "
            f"{segment['start_frame']}-{segment['end_frame']} "
            f"({segment['start_s']:.2f}-{segment['end_s']:.2f} s)"
        )

    segment_lookup = SegmentLookup(segments)

    # --------------------------------------------------
    # Video writer + CSV
    # --------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        fourcc,
        fps,
        (width, height),
    )

    csv_file = open(OUTPUT_CSV, "w", newline="")
    csv_writer = csv.writer(csv_file)

    csv_writer.writerow([
        "frame",
        "time",
        "region_id",
        "student_id",
        "yaw",
        "pitch",
        "roll",
        "left_ear",
        "right_ear",
        "avg_ear",
        "gaze",
        "shoulder_tilt",
        "head_drop",
        "head_offset",
        "hand_raised",
        "hands_near_face",
    ])

    # --------------------------------------------------
    # Main loop
    # --------------------------------------------------

    frame_number = 0
    # Global student ID management
    global_student_id = 0
    local_to_global_id = {}

    try:

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_number += 1

            # Frame 1 -> t = 0
            timestamp = (frame_number - 1) / fps

            # `frame` stays CLEAN (all crops / models use it).
            # Everything we draw goes on `annotated`.
            annotated = frame.copy()
            # --------------------------------------------------
            # PTZ Region Handling
            # --------------------------------------------------

            region_id = segment_lookup.region_id(frame_number)

            # New settled view -> start with clean tracking state
            if segment_lookup.starts_segment(frame_number):
                print(
                    f"\nSettled view (region {region_id}) "
                    f"starts at frame {frame_number}"
                )

                if USE_POSE_ONLY:
                    body_pose.reset_tracker()
                else:
                    detector.reset_tracker()
                student_manager.reset()
                angle_smoother.reset()
                local_to_global_id.clear()

            # Skip student analysis outside stationary recording regions
            if region_id is None:

                writer.write(annotated)

                if SHOW_WINDOW:
                    cv2.imshow(
                        "Classroom Engagement - Video Pipeline",
                        annotated
                    )

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                continue
            # --------------------------------------------------
            # Detection + Tracking
            # --------------------------------------------------

            pose_by_local_id = {}

            if USE_POSE_ONLY:
                raw_tracks, pose_by_local_id = body_pose.track(frame)
                results = [None]
            else:
                results = detector.detect(frame)

            if len(results) == 0:

                writer.write(annotated)

                if SHOW_WINDOW:
                    cv2.imshow(
                        "Classroom Engagement - Video Pipeline",
                        annotated,
                    )

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                continue

            if not USE_POSE_ONLY:
                raw_tracks = list(tracker.process(results[0]))

            tracks = []

            pose_matches = {}

            for local_id, bbox in raw_tracks:

                if local_id not in local_to_global_id:
                    global_student_id += 1
                    local_to_global_id[local_id] = global_student_id

                global_id = local_to_global_id[local_id]

                tracks.append((global_id, bbox))

                if local_id in pose_by_local_id:
                    pose_matches[global_id] = pose_by_local_id[local_id]

            # --------------------------------------------------
            # Two-model path: body pose (once per frame) + matching
            # --------------------------------------------------

            if not USE_POSE_ONLY:

                pose_results = body_pose.estimate(frame)

                pose_persons = body_pose.extract_keypoints(pose_results)

                pose_matches = match_poses(tracks, pose_persons)

            # --------------------------------------------------
            # Process every tracked student
            # --------------------------------------------------

            for track_id, bbox in tracks:

                student_manager.update_student(
                    track_id, bbox, frame_number, timestamp
                )

                student = student_manager.get_students()[track_id]

                x1, y1, x2, y2 = map(int, bbox)

                # --------------------------------------------------
                # Draw tracking box
                # --------------------------------------------------

                cv2.rectangle(
                    annotated,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                # --------------------------------------------------
                # Initialize measurements
                # --------------------------------------------------

                yaw = pitch = roll = None

                left_ear = right_ear = avg_ear = None

                eye_state = None
                gaze_direction = None

                shoulder_tilt = None
                head_drop = None
                head_offset = None
                hand_raised = None
                hands_near_face = None

                # --------------------------------------------------
                # Head ROI -> square ROI (from the CLEAN frame)
                # --------------------------------------------------

                _, head_bbox = extract_head_roi(frame, bbox)

                square_roi, square_bbox = make_square_roi(
                    frame,
                    head_bbox,
                )

                landmarks = None

                if square_roi is not None and square_roi.size > 0:
                    landmarks = face_mesh.detect(square_roi)

                # --------------------------------------------------
                # Face analysis
                # --------------------------------------------------

                if landmarks is not None:

                    # Landmarks are normalised to the SQUARE crop, so they
                    # must be mapped back with the SQUARE bbox.
                    landmark_points = face_mesh.get_landmark_points(
                        landmarks,
                        square_bbox,
                    )

                    student.landmark_points = landmark_points

                    # ----------------------------------------------
                    # Head pose
                    # ----------------------------------------------

                    pose = head_pose.estimate(
                        landmark_points,
                        frame.shape,
                    )

                    if pose is not None:

                        if SMOOTH_HEAD_POSE:
                            pose = angle_smoother.update(track_id, pose)

                        yaw, pitch, roll = pose

                        student.head_pose = pose

                    # ----------------------------------------------
                    # Eye analysis (does not need head pose)
                    # ----------------------------------------------

                    eye_state, left_ear, right_ear = (
                        eye_analyzer.analyse(landmark_points)
                    )

                    if eye_state is not None:

                        avg_ear = (left_ear + right_ear) / 2

                        student.left_ear = left_ear
                        student.right_ear = right_ear
                        student.avg_ear = avg_ear

                        # ------------------------------------------
                        # Gaze (needs head pose and eye state)
                        # ------------------------------------------

                        if pose is not None:

                            gaze_direction = gaze_estimator.estimate(
                                landmark_points,
                                pose,
                            )

                            student.gaze_direction = gaze_direction

                    # ----------------------------------------------
                    # Draw face landmarks on the annotated copy
                    # ----------------------------------------------

                    face_mesh.draw_landmarks(
                        annotated,
                        landmark_points,
                    )

                # --------------------------------------------------
                # Body posture
                # --------------------------------------------------

                matched_person = pose_matches.get(track_id)

                if matched_person is not None:

                    named_keypoints = body_pose.get_named_keypoints(
                        matched_person
                    )

                    posture = posture_analyzer.analyse(named_keypoints)

                    if posture is not None:

                        shoulder_tilt = posture["shoulder_tilt"]
                        head_drop = posture["head_drop"]
                        head_offset = posture["head_offset"]
                        hand_raised = posture["hand_raised"]
                        hands_near_face = posture["hands_near_face"]

                # --------------------------------------------------
                # CSV logging
                # --------------------------------------------------

                csv_writer.writerow([
                    frame_number,
                    round(timestamp, 3),
                    region_id,
                    track_id,

                    fmt(yaw, 3),
                    fmt(pitch, 3),
                    fmt(roll, 3),

                    fmt(left_ear, 4),
                    fmt(right_ear, 4),
                    fmt(avg_ear, 4),

                    gaze_direction,

                    fmt(shoulder_tilt, 3),
                    fmt(head_drop, 3),
                    fmt(head_offset, 3),
                    hand_raised,
                    hands_near_face,
                ])

                # --------------------------------------------------
                # Annotation text
                # --------------------------------------------------

                lines = [f"ID: {track_id}"]

                if yaw is not None:
                    lines.append(f"Yaw: {yaw:.1f}")

                if pitch is not None:
                    lines.append(f"Pitch: {pitch:.1f}")

                if roll is not None:
                    lines.append(f"Roll: {roll:.1f}")

                if avg_ear is not None:
                    lines.append(f"EAR: {avg_ear:.3f}")

                if gaze_direction is not None:
                    lines.append(f"Gaze: {gaze_direction}")

                if head_drop is not None:
                    lines.append(f"HeadDrop: {head_drop:.2f}")

                if shoulder_tilt is not None:
                    lines.append(f"ShTilt: {shoulder_tilt:.1f}")

                if hand_raised:
                    lines.append("HAND RAISED")

                if hands_near_face:
                    lines.append("Hand@face")

                info_y = max(20, y1 - 10 - 18 * (len(lines) - 1))

                for i, text in enumerate(lines):

                    cv2.putText(
                        annotated,
                        text,
                        (x1, info_y + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 255, 255),
                        1,
                    )

            # --------------------------------------------------
            # Write + display
            # --------------------------------------------------

            writer.write(annotated)

            if SHOW_WINDOW:

                cv2.imshow(
                    "Classroom Engagement - Video Pipeline",
                    annotated,
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if frame_number % 25 == 0:
                print(f"Processed {frame_number}/{total_frames} frames")

    finally:

        # --------------------------------------------------
        # Cleanup (runs even if the loop crashes)
        # --------------------------------------------------

        cap.release()
        writer.release()
        csv_file.close()

        cv2.destroyAllWindows()

    print()
    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Annotated video : {OUTPUT_VIDEO}")
    print(f"Student data    : {OUTPUT_CSV}")
    print("=" * 60)


if __name__ == "__main__":
    main()