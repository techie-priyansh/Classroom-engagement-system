"""
Eye features per student and per settled segment, from the pipeline CSV.

Replaces the fixed "EAR < 0.22 means closed" rule, which does not work:
  * every student has a different normal EAR (0.19 - 0.31 on the test clip),
  * looking down while writing lowers EAR (correlation up to -0.9 with pitch),
    so a fixed threshold calls writing students "closed".

What is done instead:
  1. Only frames where the eyes can be judged are used ("usable"): EAR was
     measured and the head is not turned or tilted down far.
  2. Each student gets a BASELINE EAR (75th percentile of their usable
     frames in that segment) and EAR is divided by it.
  3. "Closed" = ratio below CLOSED_RATIO for at least MIN_CLOSED_FRAMES
     frames in a row (single-frame dips are landmark noise).
  4. A closed run is a BLINK if it lasts up to BLINK_MAX_SECONDS and a LONG
     CLOSURE beyond LONG_CLOSURE_SECONDS.
  5. Per segment: PERCLOS (share of usable frames closed), blink count,
     longest closure and COVERAGE (share of frames that were usable).
     With low coverage the result is "unknown", never "eyes open".

Important limits (see the notes printed by the CLI):
  * A front camera cannot see the eyes of a student who looks down; those
    frames are excluded, so coverage is low for them.
  * Eyes cast down with lowered lids can look like a short closure.
  * The thresholds are starting points. They were checked for plausibility
    (blink rate) on one clip, NOT against labelled drowsy examples.

Run from the repository root:

    python src/eye_features.py results/video_test/student_data1.csv
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


# --------------------------------------------------
# Settings
# --------------------------------------------------

PITCH_MAX_DEG = 20.0        # looking down more than this: eyes not judged
YAW_MAX_DEG = 40.0          # head turned more than this: eyes not judged

BASELINE_PERCENTILE = 75    # "normal open eye" level of the student
MIN_BASELINE_FRAMES = 30    # usable frames needed to trust the baseline

CLOSED_RATIO = 0.6          # closed when EAR / baseline is below this
MIN_CLOSED_FRAMES = 2       # ... for at least this many frames in a row

BLINK_MAX_SECONDS = 0.5     # closed runs up to this long are blinks
LONG_CLOSURE_SECONDS = 0.5  # closed runs longer than this are long closures

MIN_COVERAGE = 0.5          # below this share of usable frames: "unknown"

SPLIT_GAP_FRAMES = 5        # a gap this long starts a new segment


# --------------------------------------------------
# Core functions (no file access, easy to test)
# --------------------------------------------------

def closed_runs(mask):
    """(start, end) index pairs (inclusive) of the True runs in a bool array."""

    runs = []
    start = None

    for i, value in enumerate(mask):

        if value and start is None:
            start = i

        elif not value and start is not None:
            runs.append((start, i - 1))
            start = None

    if start is not None:
        runs.append((start, len(mask) - 1))

    return runs


def closed_frames(ear, pitch, yaw):
    """
    Per-frame eye state of ONE student over one visit.

    Returns (usable, closed, baseline):
        usable    bool array, frames where the eyes can be judged
        closed    bool array, usable frames inside a closed run of at least
                  MIN_CLOSED_FRAMES frames (single-frame dips are ignored)
        baseline  the student's normal EAR, or None if there are too few
                  usable frames (then `closed` is all False)
    """

    ear = np.asarray(ear, dtype=np.float64)
    pitch = np.asarray(pitch, dtype=np.float64)
    yaw = np.asarray(yaw, dtype=np.float64)

    usable = (
        ~np.isnan(ear)
        & ~np.isnan(pitch) & (np.abs(pitch) <= PITCH_MAX_DEG)
        & ~np.isnan(yaw) & (np.abs(yaw) <= YAW_MAX_DEG)
    )

    closed = np.zeros(len(ear), dtype=bool)

    if usable.sum() < MIN_BASELINE_FRAMES:
        return usable, closed, None

    baseline = float(np.percentile(ear[usable], BASELINE_PERCENTILE))

    if baseline <= 0:
        return usable, closed, None

    below = usable & (ear / baseline < CLOSED_RATIO)

    for a, b in closed_runs(below):
        if b - a + 1 >= MIN_CLOSED_FRAMES:
            closed[a:b + 1] = True

    return usable, closed, baseline


def eye_features(ear, pitch, yaw, fps):
    """
    Features for ONE student in ONE segment.

    Args:
        ear, pitch, yaw: arrays with NaN where a value is missing
        fps: frames per second of the video

    Returns:
        dict (see keys below)
    """

    ear = np.asarray(ear, dtype=np.float64)
    pitch = np.asarray(pitch, dtype=np.float64)
    yaw = np.asarray(yaw, dtype=np.float64)

    frames = len(ear)

    usable, closed, baseline = closed_frames(ear, pitch, yaw)

    head_down = ~np.isnan(pitch) & (pitch > PITCH_MAX_DEG)

    coverage = float(usable.mean()) if frames else 0.0

    result = {
        "frames": frames,
        "coverage": round(coverage, 3),
        "head_down_fraction": round(float(head_down.mean()), 3) if frames else 0.0,
        "baseline_ear": None,
        "mean_ear_ratio": None,
        "perclos": None,
        "blink_count": None,
        "blinks_per_min": None,
        "longest_closure_s": None,
        "long_closure": None,
        "eyes_status": "unknown",
    }

    if baseline is None:
        return result

    ratio = ear / baseline

    runs = closed_runs(closed)

    durations = [(b - a + 1) / fps for a, b in runs]

    blinks = sum(1 for d in durations if d <= BLINK_MAX_SECONDS)

    usable_seconds = usable.sum() / fps

    longest = max(durations) if durations else 0.0

    result.update({
        "baseline_ear": round(baseline, 3),
        "mean_ear_ratio": round(float(np.mean(ratio[usable])), 3),
        "perclos": round(float(closed[usable].mean()), 3),
        "blink_count": blinks,
        "blinks_per_min": round(blinks / usable_seconds * 60.0, 1),
        "longest_closure_s": round(longest, 2),
        "long_closure": int(longest > LONG_CLOSURE_SECONDS),
        "eyes_status": "ok" if coverage >= MIN_COVERAGE else "low_coverage",
    })

    return result


def rolling_perclos(windows, n=6):
    """
    Drowsiness indicator over several consecutive windows of the SAME
    student: mean PERCLOS of the last `n` windows that have a value.

    One 7-8 s window is too short to say "drowsy"; the indicator only makes
    sense over about a minute, and only when the same student can be
    followed across camera visits (seat-based identity).

    Args:
        windows: list of eye_features() dicts in time order
    Returns:
        list of floats / None, one per window
    """

    out = []
    history = []

    for w in windows:

        if w["perclos"] is not None and w["eyes_status"] == "ok":
            history.append(w["perclos"])

        recent = history[-n:]

        out.append(round(float(np.mean(recent)), 3) if recent else None)

    return out


# --------------------------------------------------
# CSV processing
# --------------------------------------------------

def _float(value):
    return float(value) if value not in ("", None) else np.nan


def split_segments(rows):
    """Split one student's rows (sorted by frame) wherever frames jump."""

    segments = []
    current = []

    for row in rows:

        if current and int(row["frame"]) - int(current[-1]["frame"]) > SPLIT_GAP_FRAMES:
            segments.append(current)
            current = []

        current.append(row)

    if current:
        segments.append(current)

    return segments


def process_csv(csv_path):
    """Return (list of per-segment dicts, fps)."""

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    by_student = defaultdict(list)

    for row in rows:
        by_student[(row["region_id"], row["student_id"])].append(row)

    # fps from the time column
    times = sorted({(int(r["frame"]), float(r["time"])) for r in rows})
    deltas = [
        (t2 - t1) / (f2 - f1)
        for (f1, t1), (f2, t2) in zip(times, times[1:])
        if f2 - f1 == 1 and t2 > t1
    ]
    fps = 1.0 / float(np.median(deltas)) if deltas else 25.0

    output = []

    for (region, student), student_rows in by_student.items():

        student_rows.sort(key=lambda r: int(r["frame"]))

        for segment in split_segments(student_rows):

            features = eye_features(
                [_float(r["avg_ear"]) for r in segment],
                [_float(r["pitch"]) for r in segment],
                [_float(r["yaw"]) for r in segment],
                fps,
            )

            first, last = segment[0], segment[-1]

            output.append({
                "region_id": region,
                "student_id": student,
                "start_frame": int(first["frame"]),
                "end_frame": int(last["frame"]),
                "start_s": float(first["time"]),
                "end_s": float(last["time"]),
                **features,
            })

    output.sort(key=lambda d: (d["start_frame"], int(d["student_id"])))

    return output, fps


def main():

    csv_path = Path(sys.argv[1] if len(sys.argv) > 1
                    else "results/video_test/student_data1.csv")

    out_path = (
        Path(sys.argv[2]) if len(sys.argv) > 2
        else csv_path.with_name("student_windows.csv")
    )

    windows, fps = process_csv(csv_path)

    with open(out_path, "w", newline="") as f:

        writer = csv.DictWriter(f, fieldnames=list(windows[0].keys()))
        writer.writeheader()
        writer.writerows(windows)

    ok = [w for w in windows if w["eyes_status"] == "ok"]

    print(f"{len(windows)} student segments ({fps:.0f} fps) -> {out_path}")
    print(f"  eyes judged (coverage >= {MIN_COVERAGE:.0%}): {len(ok)}")
    print(f"  low coverage / unknown                 : {len(windows) - len(ok)}")

    if ok:
        print(
            "  blinks per minute (median)             : "
            f"{np.median([w['blinks_per_min'] for w in ok]):.1f}"
        )
        print(
            "  segments with a closure > "
            f"{LONG_CLOSURE_SECONDS:.1f} s               : "
            f"{sum(w['long_closure'] for w in ok)} of {len(ok)}"
        )

    print()
    print("Note: thresholds are starting points, checked for plausibility only.")
    print("Eyes cast down with lowered lids can look like a short closure.")


if __name__ == "__main__":
    main()
