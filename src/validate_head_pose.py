"""
Multi-student, multi-frame validation of head pose estimation.

For every tracked student in many frames it measures:
  * how often a face is found and a head pose is solved
  * reprojection error of the solved pose
  * invalid solutions (behind the camera, solver failure, bad eye distance)
  * temporal stability (frame-to-frame change, sudden jumps, raw vs smoothed)

Run from the repository root:

    python src/validate_head_pose.py
    python src/validate_head_pose.py --video tests/test2.mp4 --ranges 0-300
    python src/validate_head_pose.py --step 2          # every 2nd frame

Results go to results/head_pose_validation/ (per-frame CSV, per-student
CSV and a text summary).
"""

from paths import TEST_VIDEO, RESULTS_DIR
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

from detector import Detector
from tracker import Tracker
from face_mesh import FaceMeshDetector
from head_pose import HeadPoseEstimator, AngleSmoother
from utils import extract_head_roi, make_square_roi


# --------------------------------------------------
# Defaults
# --------------------------------------------------

DEFAULT_VIDEO = str(TEST_VIDEO)
DEFAULT_RANGES = "50-249,300-499,550-749"     # same as video_pipeline.py
DEFAULT_OUT = str(RESULTS_DIR / "head_pose_validation")

EMA_ALPHA = 0.4          # same smoothing as video_pipeline.py
JUMP_DEG = 30.0          # a frame-to-frame change above this is a "jump"
WRAP_DEG = 150.0         # |pitch| or |roll| above this suggests angle wrap

# Pass / check limits used in the summary
MIN_POSE_RATE = 0.95     # of faces found
MAX_REPROJ_P95_PX = 8.0
MAX_JUMP_RATE = 0.01     # of consecutive frame pairs


# --------------------------------------------------
# Collecting measurements
# --------------------------------------------------

def parse_ranges(text):
    """'50-249,300-499' -> [(50, 249), (300, 499)] (1-based frame numbers)."""

    ranges = []

    for part in text.split(","):
        start, end = part.split("-")
        ranges.append((int(start), int(end)))

    return ranges


def region_of(frame_number, ranges):
    """1-based index of the range containing the frame, or None."""

    for index, (start, end) in enumerate(ranges, start=1):
        if start <= frame_number <= end:
            return index

    return None


def collect(video_path, ranges, step):
    """Run detection, tracking, face mesh and head pose; return records."""

    detector = Detector()
    tracker = Tracker()
    face_mesh = FaceMeshDetector()
    head_pose = HeadPoseEstimator()

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    records = []
    previous_region = None
    frame_number = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        region = region_of(frame_number, ranges)

        if region is None:
            previous_region = None
            continue

        # new region -> new tracker state (same as the main pipeline)
        if region != previous_region:
            detector.reset_tracker()
            previous_region = region

        # Tracking needs consecutive frames, so every frame goes through
        # the detector; --step only thins out what is measured below.
        results = detector.detect(frame)

        if len(results) == 0:
            continue

        tracks = tracker.process(results[0])

        if (frame_number - ranges[region - 1][0]) % step:
            continue

        for track_id, bbox in tracks:

            record = {
                "frame": frame_number,
                "region": region,
                "track_id": track_id,
                "face_found": 0,
                "pose_ok": 0,
                "failure": "",
                "yaw": None,
                "pitch": None,
                "roll": None,
                "reproj_px": None,
                "tz_mm": None,
            }

            _, head_bbox = extract_head_roi(frame, bbox)
            square_roi, square_bbox = make_square_roi(frame, head_bbox)

            landmarks = None

            if square_roi is not None and square_roi.size > 0:
                landmarks = face_mesh.detect(square_roi)

            if landmarks is None:
                record["failure"] = "no_face"
                records.append(record)
                continue

            record["face_found"] = 1

            points = face_mesh.get_landmark_points(landmarks, square_bbox)

            result = head_pose.estimate_full(points, frame.shape)

            if result is None:
                record["failure"] = head_pose.last_failure or "unknown"
                records.append(record)
                continue

            record["pose_ok"] = 1
            record["yaw"] = result["yaw"]
            record["pitch"] = result["pitch"]
            record["roll"] = result["roll"]
            record["reproj_px"] = float(result["reprojection_error"])
            record["tz_mm"] = float(result["tvec"][2, 0])

            records.append(record)

    cap.release()

    return records, fps


# --------------------------------------------------
# Analysis
# --------------------------------------------------

def percentile(values, q):
    return float(np.percentile(values, q)) if len(values) else float("nan")


def stability(records, step):
    """
    Frame-to-frame behaviour of the angles, raw and EMA-smoothed.

    Only consecutive recorded frames of the same student are compared
    (a gap in detection breaks the pair).
    """

    smoother = AngleSmoother(alpha=EMA_ALPHA)

    by_student = {}

    for r in records:
        by_student.setdefault((r["region"], r["track_id"]), []).append(r)

    raw_diffs = []
    smooth_diffs = []

    for key, rows in by_student.items():

        rows.sort(key=lambda r: r["frame"])

        previous = None
        previous_smooth = None

        for r in rows:

            if not r["pose_ok"]:
                previous = None
                previous_smooth = None
                smoother._state.pop(key, None)
                continue

            raw = np.array([r["yaw"], r["pitch"], r["roll"]])
            smooth = np.array(smoother.update(key, raw))

            if previous is not None and r["frame"] - previous["frame"] == step:
                raw_diffs.append(np.abs(raw - previous["raw"]))
                smooth_diffs.append(np.abs(smooth - previous_smooth))

            previous = {"frame": r["frame"], "raw": raw}
            previous_smooth = smooth

    return np.array(raw_diffs), np.array(smooth_diffs)


def student_table(records):
    """One row per (region, track) with its own success and error numbers."""

    by_student = {}

    for r in records:
        by_student.setdefault((r["region"], r["track_id"]), []).append(r)

    rows = []

    for (region, track_id), items in sorted(by_student.items()):

        frames = len(items)
        faces = sum(r["face_found"] for r in items)
        poses = sum(r["pose_ok"] for r in items)

        reproj = [r["reproj_px"] for r in items if r["pose_ok"]]
        yaws = [r["yaw"] for r in items if r["pose_ok"]]

        rows.append({
            "region": region,
            "track_id": track_id,
            "frames": frames,
            "face_rate": round(faces / frames, 3),
            "pose_rate": round(poses / frames, 3),
            "reproj_median_px": round(float(np.median(reproj)), 2) if reproj else "",
            "reproj_p95_px": round(percentile(reproj, 95), 2) if reproj else "",
            "yaw_std": round(float(np.std(yaws)), 1) if yaws else "",
        })

    return rows


def summarise(records, step=1):
    """Return (list of text lines, dict of key numbers)."""

    lines = []
    numbers = {}

    total = len(records)

    if total == 0:
        return ["No student-frames were recorded."], numbers

    faces = sum(r["face_found"] for r in records)
    poses = sum(r["pose_ok"] for r in records)

    failures = {}
    for r in records:
        if r["failure"]:
            failures[r["failure"]] = failures.get(r["failure"], 0) + 1

    reproj = [r["reproj_px"] for r in records if r["pose_ok"]]
    tz = [r["tz_mm"] for r in records if r["pose_ok"]]

    pose_rate_of_faces = poses / faces if faces else 0.0

    lines.append(f"Student-frames analysed     : {total}")
    lines.append(f"Face found                  : {faces} ({faces / total:.1%})")
    lines.append(
        f"Head pose solved            : {poses} "
        f"({pose_rate_of_faces:.1%} of faces, {poses / total:.1%} of all)"
    )
    lines.append(f"Failures                    : {failures or 'none'}")

    if reproj:
        lines.append(
            "Reprojection error (px)     : "
            f"median {np.median(reproj):.2f} | p95 {percentile(reproj, 95):.2f} "
            f"| max {max(reproj):.2f}"
        )
        lines.append(
            "Distance to camera (mm)     : "
            f"median {np.median(tz):.0f} | min {min(tz):.0f} | max {max(tz):.0f}"
        )

    yaw = [r["yaw"] for r in records if r["pose_ok"]]
    pitch = [r["pitch"] for r in records if r["pose_ok"]]
    roll = [r["roll"] for r in records if r["pose_ok"]]

    wrap = sum(
        1 for p, ro in zip(pitch, roll)
        if abs(p) > WRAP_DEG or abs(ro) > WRAP_DEG
    )

    lines.append(
        f"Angle ranges (deg)          : yaw {min(yaw):.0f}..{max(yaw):.0f} | "
        f"pitch {min(pitch):.0f}..{max(pitch):.0f} | "
        f"roll {min(roll):.0f}..{max(roll):.0f}"
    )
    lines.append(
        f"Possible angle wrap (|x|>{WRAP_DEG:.0f}) : {wrap} frames"
    )

    raw_diffs, smooth_diffs = stability(records, step)

    jump_rate = float("nan")

    if len(raw_diffs):

        jumps = int(np.sum(np.any(raw_diffs > JUMP_DEG, axis=1)))
        jump_rate = jumps / len(raw_diffs)

        lines.append(f"Consecutive frame pairs     : {len(raw_diffs)}")

        for name, index in (("yaw", 0), ("pitch", 1), ("roll", 2)):
            lines.append(
                f"  {name:5s} frame-to-frame change (deg): "
                f"raw median {np.median(raw_diffs[:, index]):.2f} "
                f"p95 {percentile(raw_diffs[:, index], 95):.1f} | "
                f"smoothed median {np.median(smooth_diffs[:, index]):.2f} "
                f"p95 {percentile(smooth_diffs[:, index], 95):.1f}"
            )

        lines.append(
            f"Jumps > {JUMP_DEG:.0f} deg between frames  : {jumps} "
            f"({jump_rate:.2%} of pairs)"
        )

    # ---- verdict --------------------------------------------------------

    p95 = percentile(reproj, 95) if reproj else float("nan")

    checks = [
        (f"pose solved >= {MIN_POSE_RATE:.0%} of faces",
         pose_rate_of_faces >= MIN_POSE_RATE),
        (f"reprojection p95 <= {MAX_REPROJ_P95_PX:.0f} px",
         p95 <= MAX_REPROJ_P95_PX),
        ("no solutions behind the camera",
         failures.get("behind_camera", 0) == 0),
        (f"jumps <= {MAX_JUMP_RATE:.0%} of frame pairs",
         jump_rate <= MAX_JUMP_RATE),
    ]

    lines.append("")
    lines.append("Checks (limits are starting points, adjust with more data):")

    for text, passed in checks:
        lines.append(f"  [{'OK' if passed else 'CHECK'}] {text}")

    numbers.update({
        "pose_rate_of_faces": pose_rate_of_faces,
        "reproj_p95": p95,
        "jump_rate": jump_rate,
    })

    return lines, numbers


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--video", default=DEFAULT_VIDEO)
    parser.add_argument(
        "--ranges", default=DEFAULT_RANGES,
        help="1-based frame ranges to analyse, e.g. 50-249,300-499",
    )
    parser.add_argument(
        "--step", type=int, default=1,
        help="record every Nth frame (tracking still sees every frame)",
    )
    parser.add_argument("--out", default=DEFAULT_OUT)

    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ranges = parse_ranges(args.ranges)

    print(f"Video  : {args.video}")
    print(f"Ranges : {ranges}  (every {args.step} frame)")

    records, fps = collect(args.video, ranges, args.step)

    lines, _ = summarise(records, args.step)

    print()
    print("\n".join(lines))

    # ---- files ----------------------------------------------------------

    with open(out_dir / "frames.csv", "w", newline="") as f:

        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    table = student_table(records)

    with open(out_dir / "students.csv", "w", newline="") as f:

        writer = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        writer.writeheader()
        writer.writerows(table)

    with open(out_dir / "summary.txt", "w") as f:
        f.write("\n".join(lines) + "\n")

    print()
    print(f"Saved to {out_dir}/ (frames.csv, students.csv, summary.txt)")


if __name__ == "__main__":
    sys.exit(main())
