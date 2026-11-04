"""
How far does the camera view move between visits to the same region?

Give it two videos (or the same video twice) and the time, in seconds, of a
settled frame of the same region in each. It prints the shift, zoom and
rotation of the second view relative to the first.

    python src/compare_visits.py clipA.mp4 8.0 clipB.mp4 8.0

If the shift is only a few pixels, seat positions from the first visit can be
reused directly. If it is tens of pixels (or the zoom differs), detections
have to be aligned first (see seat_map.py).

Tip: pick a frame near the middle of the settled period of each visit
(src/settle.py prints the settled segments).
"""

import sys

import cv2

from seat_map import estimate_view_shift


def read_frame(path, seconds):

    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {path}")

    cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000.0)

    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError(f"No frame at {seconds}s in {path}")

    return frame


def main():

    if len(sys.argv) != 5:
        print(__doc__)
        return 1

    path_a, time_a, path_b, time_b = sys.argv[1:]

    frame_a = read_frame(path_a, float(time_a))
    frame_b = read_frame(path_b, float(time_b))

    if frame_a.shape != frame_b.shape:
        print(f"Different resolutions: {frame_a.shape} vs {frame_b.shape}")
        return 1

    result = estimate_view_shift(frame_a, frame_b)

    print(f"Matched points : {result['matches']} ({result['inliers']} agree)")

    if result["M"] is None:
        print("Could not estimate a shift: the views do not match.")
        return 1

    print(f"Shift          : dx {result['dx']:+.1f} px, dy {result['dy']:+.1f} px")
    print(f"Zoom           : x{result['scale']:.4f}")
    print(f"Rotation       : {result['rotation_deg']:+.2f} deg")
    print(f"Reliable       : {'yes' if result['ok'] else 'NO - too few matching points'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
