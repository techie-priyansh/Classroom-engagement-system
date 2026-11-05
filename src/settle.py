"""
Find the frames where the PTZ camera is standing still and in focus.

The camera sweeps Region 1 -> 2 -> 3 -> 1 ... Between regions it moves
(about 2-3 s) and the picture is blurred, and after it stops the autofocus
needs a moment. Student analysis must only use the settled frames.

Two image measurements per frame (on a small grey copy of the frame):
  motion    mean absolute difference to the previous frame
            (large while the camera moves, close to 0 when it is still)
  sharpness variance of the Laplacian
            (drops while moving or out of focus)

A SEGMENT is a run of still frames, trimmed so that it starts only after the
sharpness has recovered and ends a moment before the next move.

Optionally the recorder's sweep log can be used instead (see
`segments_from_sweep_log`); the image measurements then only trim the start
of each logged hold window.

Usage:
    segments = find_settled_segments("tests/clip.mp4")
    # [{"region": 1, "start_frame": 88, "end_frame": 262, ...}, ...]
"""

from paths import TEST_VIDEO
import json
from pathlib import Path

import cv2
import numpy as np


# --------------------------------------------------
# Settings
# --------------------------------------------------

ANALYSIS_SIZE = (480, 270)   # frames are shrunk to this before measuring

MOTION_THRESHOLD = 1.5       # mean abs grey-level change per frame; moving above
                             # (measured: still 0.01-0.5, moving 3.7-11)
MOTION_DILATE_FRAMES = 2     # frames next to a moving frame also count as moving
MIN_STILL_SECONDS = 2.0      # shorter still periods are pauses, not regions

SHARPNESS_FRACTION = 0.90    # settled once sharpness reaches this share of the
                             # segment's own (late-half) median sharpness
SHARPNESS_HOLD_FRAMES = 5    # ... and stays there for this many frames

START_MARGIN_SECONDS = 0.2   # extra margin after focus has recovered
END_MARGIN_SECONDS = 0.2     # stop this long before the next move starts

MIN_SEGMENT_SECONDS = 2.0    # drop segments shorter than this after trimming

NUM_REGIONS = 3              # the sweep repeats this many regions in order


# --------------------------------------------------
# Measuring
# --------------------------------------------------

def measure_video(video_path):
    """
    One pass over the video.

    Returns:
        fps, motion (array), sharpness (array); index 0 = frame 1.
        motion[0] is 0 (there is no previous frame).
    """

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 0:
        fps = 25.0

    motion = []
    sharpness = []
    previous = None

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        grey = cv2.cvtColor(
            cv2.resize(frame, ANALYSIS_SIZE, interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2GRAY,
        )

        motion.append(
            0.0 if previous is None
            else float(np.mean(cv2.absdiff(grey, previous)))
        )

        sharpness.append(float(cv2.Laplacian(grey, cv2.CV_64F).var()))

        previous = grey

    cap.release()

    return fps, np.array(motion), np.array(sharpness)


# --------------------------------------------------
# Segmenting
# --------------------------------------------------

def _runs(mask):
    """Start/end (inclusive) index pairs of the True runs in a bool array."""

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


def segments_from_measurements(fps, motion, sharpness):
    """
    Turn per-frame motion and sharpness into settled segments.

    Returns a list of dicts with 1-based frame numbers (like the rest of the
    pipeline), times in seconds and a region number following the sweep order.
    """

    moving = motion > MOTION_THRESHOLD

    # frames right next to a moving frame are blurred too
    k = MOTION_DILATE_FRAMES
    moving = np.convolve(moving.astype(int), np.ones(2 * k + 1), "same") > 0

    min_still = int(MIN_STILL_SECONDS * fps)
    start_margin = int(round(START_MARGIN_SECONDS * fps))
    end_margin = int(round(END_MARGIN_SECONDS * fps))
    min_segment = int(MIN_SEGMENT_SECONDS * fps)

    segments = []

    for run_start, run_end in _runs(~moving):

        if run_end - run_start + 1 < min_still:
            continue

        # Sharpness of this view once the focus has settled: use the
        # second half of the still period as the reference.
        half = run_start + (run_end - run_start) // 2
        reference = float(np.median(sharpness[half:run_end + 1]))

        sharp_enough = sharpness >= SHARPNESS_FRACTION * reference

        start = None

        for i in range(run_start, run_end - SHARPNESS_HOLD_FRAMES + 2):

            if np.all(sharp_enough[i:i + SHARPNESS_HOLD_FRAMES]):
                start = i
                break

        if start is None:
            continue

        # a run that is already sharp from its first frame (e.g. a video
        # that starts in a settled view) needs no start margin
        if start > run_start:
            start += start_margin

        # a run that touches the end of the video has no following move
        end = run_end if run_end == len(moving) - 1 else run_end - end_margin

        if end - start + 1 < min_segment:
            continue

        segments.append({
            "start_frame": int(start) + 1,
            "end_frame": int(end) + 1,
        })

    return _finish(segments, fps)


def _finish(segments, fps):
    """Add times and the region number (sweep order 1, 2, 3, 1, 2, 3 ...)."""

    for index, segment in enumerate(segments):

        segment["region"] = index % NUM_REGIONS + 1
        segment["start_s"] = round((segment["start_frame"] - 1) / fps, 2)
        segment["end_s"] = round(segment["end_frame"] / fps, 2)

    return segments


def find_settled_segments(video_path, cache_path=None):
    """
    Settled segments of a video. If `cache_path` is given and exists, it is
    loaded instead of measuring the video again; otherwise the result is
    saved there.
    """

    if cache_path is not None and Path(cache_path).exists():
        return load_segments(cache_path)

    fps, motion, sharpness = measure_video(video_path)

    segments = segments_from_measurements(fps, motion, sharpness)

    if cache_path is not None:
        save_segments(segments, cache_path)

    return segments


# --------------------------------------------------
# Sweep log from the recorder (optional)
# --------------------------------------------------

def segments_from_sweep_log(log_path, video_path):
    """
    Use the recorder's sweep log: hold windows come from the log, and the
    image measurements only trim the part of each window where the focus is
    still recovering.

    Log format (written by ptz_record_sweep.py):
        [{"round": 1, "region": 1, "hold_start_s": 12.1, "hold_end_s": 20.1}, ...]
        times are seconds from the start of the recorded video.
    """

    entries = json.loads(Path(log_path).read_text())

    fps, motion, sharpness = measure_video(video_path)

    total = len(motion)

    segments = []

    for entry in entries:

        first = max(0, int(entry["hold_start_s"] * fps))
        last = min(total - 1, int(entry["hold_end_s"] * fps) - 1)

        if last - first + 1 < int(MIN_SEGMENT_SECONDS * fps):
            continue

        window_motion = motion.copy()
        window_motion[:first] = 1e9           # outside the window = "moving"
        window_motion[last + 1:] = 1e9

        found = segments_from_measurements(fps, window_motion, sharpness)

        # keep the longest segment inside the window
        found = [s for s in found if s["start_frame"] - 1 >= first]

        if not found:
            continue

        best = max(found, key=lambda s: s["end_frame"] - s["start_frame"])

        best["region"] = int(entry["region"])
        segments.append(best)

    for segment in segments:
        segment["start_s"] = round((segment["start_frame"] - 1) / fps, 2)
        segment["end_s"] = round(segment["end_frame"] / fps, 2)

    return segments


# --------------------------------------------------
# Saving
# --------------------------------------------------

def save_segments(segments, path):

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(segments, indent=2))


def load_segments(path):

    return json.loads(Path(path).read_text())


# --------------------------------------------------
# Lookup used by the pipeline
# --------------------------------------------------

class SegmentLookup:
    """Which settled segment (if any) does a frame number belong to?"""

    def __init__(self, segments):
        self.segments = segments

    def segment_index(self, frame_number):
        """Index of the segment containing the frame, or None."""

        for index, s in enumerate(self.segments):
            if s["start_frame"] <= frame_number <= s["end_frame"]:
                return index

        return None

    def region_id(self, frame_number):

        index = self.segment_index(frame_number)

        return None if index is None else self.segments[index]["region"]

    def starts_segment(self, frame_number):
        """True on the first frame of a segment (reset trackers here)."""

        return any(s["start_frame"] == frame_number for s in self.segments)


if __name__ == "__main__":

    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else str(TEST_VIDEO)

    for s in find_settled_segments(path):
        print(
            f"Region {s['region']}: frames {s['start_frame']}-{s['end_frame']} "
            f"({s['start_s']:.2f}-{s['end_s']:.2f} s, "
            f"{s['end_s'] - s['start_s']:.1f} s)"
        )
