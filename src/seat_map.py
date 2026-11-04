"""
Seat map and view alignment: PROTOTYPE for re-identifying students when the
camera comes back to a region.

The camera sees each region for about 7 s every 30 s. A tracker cannot bridge
that gap, but students stay in their seats. So identity is tied to the SEAT:

  1. Build a SEAT MAP for a region from the first visit (head positions).
  2. On a later visit, measure how far the camera view has moved compared with
     the reference view (`estimate_view_shift`) and move the detections back
     into the reference view.
  3. Assign every detection to the nearest free seat (`SeatMap.assign`,
     optimal one-to-one matching, with a distance limit).

STATUS: tested on one clip (one visit per region) with an artificial shift.
It has NOT been tested across real camera visits. Use `compare_visits.py` on
a second clip first: it measures how far the real camera view drifts between
visits, which decides whether this simple approach is enough.
"""

import cv2
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment


# --------------------------------------------------
# Settings
# --------------------------------------------------

FEATURE_WIDTH = 960          # frames are shrunk to this width before matching
MAX_FEATURES = 3000
RANSAC_THRESHOLD_PX = 3.0    # in the shrunk image

MIN_INLIERS = 40             # fewer matched points: the views do not match
MIN_INLIER_RATIO = 0.35

MAX_SEAT_DISTANCE_PX = 90.0  # a detection further than this from every seat
                             # stays unassigned (full-resolution pixels)


# --------------------------------------------------
# View alignment
# --------------------------------------------------

def estimate_view_shift(reference, image):
    """
    How has the camera view moved from `reference` to `image`?

    Both are BGR frames of the SAME region. The motion is estimated from the
    static scene (benches, walls) as a similarity transform (shift, zoom and
    a little rotation), which is what a small pan / tilt / zoom error is.

    Returns a dict:
        ok          True if enough matching points were found
        M           2x3 matrix mapping reference -> image, full-resolution pixels
        dx, dy      shift of the image centre in pixels
        scale       zoom factor (1.0 = same zoom)
        rotation_deg
        inliers, matches
    """

    scale_down = FEATURE_WIDTH / reference.shape[1]

    def prepare(frame):
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.resize(grey, None, fx=scale_down, fy=scale_down,
                          interpolation=cv2.INTER_AREA)

    orb = cv2.ORB_create(MAX_FEATURES)

    kp1, des1 = orb.detectAndCompute(prepare(reference), None)
    kp2, des2 = orb.detectAndCompute(prepare(image), None)

    result = {
        "ok": False, "M": None, "dx": None, "dy": None,
        "scale": None, "rotation_deg": None, "inliers": 0, "matches": 0,
    }

    if des1 is None or des2 is None:
        return result

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)

    result["matches"] = len(matches)

    if len(matches) < 10:
        return result

    src = np.float32([kp1[m.queryIdx].pt for m in matches])
    dst = np.float32([kp2[m.trainIdx].pt for m in matches])

    M_small, inlier_mask = cv2.estimateAffinePartial2D(
        src, dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_THRESHOLD_PX,
    )

    if M_small is None:
        return result

    inliers = int(inlier_mask.sum())

    # Convert the matrix from shrunk-image pixels to full-resolution pixels.
    S = np.diag([scale_down, scale_down, 1.0])
    M3 = np.vstack([M_small, [0, 0, 1]])
    M_full = (np.linalg.inv(S) @ M3 @ S)[:2]

    a, b = M_full[0, 0], M_full[1, 0]
    zoom = float(np.hypot(a, b))
    rotation = float(np.degrees(np.arctan2(b, a)))

    h, w = reference.shape[:2]
    centre = np.array([w / 2.0, h / 2.0, 1.0])
    moved = M_full @ centre

    result.update({
        "ok": bool(
            inliers >= MIN_INLIERS
            and inliers / len(matches) >= MIN_INLIER_RATIO
        ),
        "M": M_full,
        "dx": float(moved[0] - centre[0]),
        "dy": float(moved[1] - centre[1]),
        "scale": zoom,
        "rotation_deg": rotation,
        "inliers": inliers,
    })

    return result


def to_reference(points, M):
    """Move points from the current view back into the reference view."""

    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)

    if M is None:
        return points

    inverse = cv2.invertAffineTransform(np.asarray(M, dtype=np.float64))

    return points @ inverse[:, :2].T + inverse[:, 2]


# --------------------------------------------------
# Seat map
# --------------------------------------------------

class SeatMap:
    """Seats of ONE region, as head positions in the reference view."""

    def __init__(self, positions, region):
        """
        positions: (N, 2) head positions (e.g. the nose keypoint) in the
                   reference view, one per seat.
        """

        positions = np.asarray(positions, dtype=np.float64).reshape(-1, 2)

        # number the seats top-to-bottom, then left-to-right
        order = np.lexsort((positions[:, 0], np.round(positions[:, 1] / 60)))

        self.region = region
        self.positions = positions[order]
        self.seat_ids = [
            f"R{region}-S{i + 1}" for i in range(len(self.positions))
        ]

    def assign(self, positions, M=None, max_distance=MAX_SEAT_DISTANCE_PX):
        """
        Match detections to seats, one detection per seat at most.

        positions: (K, 2) head positions in the CURRENT view
        M        : view shift from estimate_view_shift() (None = no shift)

        Returns a list of K seat ids (None when no free seat is close enough).
        """

        positions = np.asarray(positions, dtype=np.float64).reshape(-1, 2)

        if len(positions) == 0:
            return []

        in_reference = to_reference(positions, M)

        cost = np.linalg.norm(
            in_reference[:, None, :] - self.positions[None, :, :], axis=2
        )

        rows, cols = linear_sum_assignment(cost)

        result = [None] * len(positions)

        for r, c in zip(rows, cols):
            if cost[r, c] <= max_distance:
                result[r] = self.seat_ids[c]

        return result


def build_seat_map(position_lists, region, min_presence=0.4, cluster_px=45.0):
    """
    Build a seat map from several frames of one visit.

    position_lists: list (one entry per frame) of lists of (x, y) head
        positions, or of dicts {track_id: (x, y)} (the IDs are ignored).

    Positions of all frames are pooled and grouped by closeness, so a
    student whose tracker ID keeps changing still gives ONE seat. A group
    becomes a seat when it appears in at least `min_presence` of the frames
    (rare ghost detections are ignored).
    """

    points = []

    for frame_positions in position_lists:

        values = (
            frame_positions.values()
            if isinstance(frame_positions, dict) else frame_positions
        )

        points.extend(tuple(v) for v in values)

    if len(points) < 2:
        return SeatMap(np.empty((0, 2)), region)

    points = np.asarray(points, dtype=np.float64)

    groups = fcluster(
        linkage(points, method="average"),
        t=cluster_px,
        criterion="distance",
    )

    needed = max(5, int(min_presence * len(position_lists)))

    centres = [
        points[groups == g].mean(axis=0)
        for g in np.unique(groups)
        if np.sum(groups == g) >= needed
    ]

    return SeatMap(np.array(centres).reshape(-1, 2), region)
