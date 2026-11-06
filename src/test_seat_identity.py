"""Tests for seat_identity.py on synthetic data (no camera, no video).

Run from the src folder:  python test_seat_identity.py
"""

import cv2
import numpy as np

from seat_identity import assign_seats, merge_seats, vote_track_seats
from visits import find_visits


# ---- a fake classroom: a textured background and 4 seats ----

rng = np.random.RandomState(3)
BACKGROUND = cv2.GaussianBlur(
    (rng.rand(1080, 1920, 3) * 255).astype(np.uint8), (0, 0), 3)
BACKGROUND = cv2.normalize(BACKGROUND, None, 0, 255, cv2.NORM_MINMAX)

SEATS = [(300, 300), (700, 320), (1100, 310), (1500, 330)]


def make_rows(visit_defs):
    """
    visit_defs: list of (first_frame, region, shift (dx, dy), {track: seat index},
                         optional id_change_at)
    The head position of a student in a visit = seat + shift (+ small noise).
    """

    rows = []

    for first, region, (dx, dy), tracks in visit_defs:

        for f in range(first, first + 60):

            for track, seat in tracks.items():

                x, y = SEATS[seat]

                rows.append({
                    "frame": str(f), "region_id": str(region),
                    "student_id": str(track),
                    "head_x": f"{x + dx + rng.randn():.1f}",
                    "head_y": f"{y + dy + rng.randn():.1f}",
                })

    return rows


def frame_for(shifts):
    """get_frame: the background moved by the shift of the visit it belongs to."""

    def get(number):
        for first, (dx, dy) in shifts.items():
            if first <= number < first + 60:
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                return cv2.warpAffine(BACKGROUND, M, (1920, 1080))
        raise KeyError(number)

    return get


def test_same_student_gets_same_seat_after_camera_drift():
    rows = make_rows([
        (1, 1, (0, 0), {1: 0, 2: 1, 3: 2, 4: 3}),
        (200, 1, (0, -25), {11: 0, 12: 1, 13: 2, 14: 3}),     # drifted 25 px up
    ])

    seats, maps, infos = assign_seats(rows, frame_for({1: (0, 0), 200: (0, -25)}))

    assert len(maps["1"].positions) == 4
    assert abs(infos[1]["dy"] + 25) < 1.0, infos[1]

    first = {r["student_id"]: s for r, s in zip(rows, seats) if int(r["frame"]) < 100}
    second = {r["student_id"]: s for r, s in zip(rows, seats) if int(r["frame"]) >= 200}

    assert len(set(first.values())) == 4 and "" not in first.values()
    for a, b in ((1, 11), (2, 12), (3, 13), (4, 14)):
        assert first[str(a)] == second[str(b)], (first, second)


def test_changed_tracker_id_keeps_the_seat():
    rows = make_rows([(1, 1, (0, 0), {1: 0, 2: 1})])

    # student on seat 0 loses ID 1 at frame 30 and continues as ID 9
    for r in rows:
        if r["student_id"] == "1" and int(r["frame"]) >= 30:
            r["student_id"] = "9"

    seats, _, _ = assign_seats(rows, frame_for({1: (0, 0)}))

    got = {(r["student_id"]): s for r, s in zip(rows, seats)}

    assert got["1"] == got["9"] != "", got


def test_short_ghost_far_from_every_seat_gets_none():
    rows = make_rows([(1, 1, (0, 0), {1: 0, 2: 1, 3: 2})])

    for f in range(1, 16):        # 15 of 60 frames: too rare to be a seat
        rows.append({"frame": str(f), "region_id": "1", "student_id": "99",
                     "head_x": "1000.0", "head_y": "800.0"})

    seats, _, _ = assign_seats(rows, frame_for({1: (0, 0)}))

    ghost = {s for r, s in zip(rows, seats) if r["student_id"] == "99"}

    assert ghost == {""}, ghost


def test_seat_empty_in_first_visit_is_still_found():
    rows = make_rows([
        (1, 1, (0, 0), {1: 0, 2: 1}),
        (200, 1, (0, 0), {11: 0, 12: 1, 13: 3}),              # a new student on seat 3
    ])

    seats, maps, _ = assign_seats(rows, frame_for({1: (0, 0), 200: (0, 0)}))

    assert len(maps["1"].positions) == 3
    new = {s for r, s in zip(rows, seats) if r["student_id"] == "13"}
    assert len(new) == 1 and "" not in new


def test_two_regions_have_their_own_seat_names():
    rows = make_rows([
        (1, 1, (0, 0), {1: 0}),
        (100, 2, (0, 0), {5: 0}),
    ])

    seats, maps, _ = assign_seats(rows, frame_for({1: (0, 0), 100: (0, 0)}))

    names = {s for s in seats}
    assert names == {"R1-S1", "R2-S1"}, names


def test_overlapping_tracks_cannot_share_a_seat():
    frames = {
        "a": {f: "R1-S1" for f in range(1, 61)},
        "b": {f: "R1-S1" for f in range(1, 31)},      # same time, same seat
    }
    result = vote_track_seats(frames)
    assert result["a"] == "R1-S1" and result["b"] is None


def test_following_tracks_may_share_a_seat():
    frames = {
        "a": {f: "R1-S1" for f in range(1, 31)},
        "b": {f: "R1-S1" for f in range(31, 61)},
    }
    result = vote_track_seats(frames)
    assert result["a"] == result["b"] == "R1-S1"


def test_track_that_flickers_between_seats_gets_none():
    frames = {"a": {f: ("R1-S1" if f % 2 else "R1-S2") for f in range(1, 21)}}
    assert vote_track_seats(frames)["a"] is not None   # 50% each -> first wins
    frames = {"a": {f: ["R1-S1", "R1-S2", "R1-S3"][f % 3] for f in range(1, 31)}}
    assert vote_track_seats(frames)["a"] is None


def test_merge_seats_joins_close_centres():
    merged = merge_seats([[(100, 100), (500, 100)], [(110, 95), (900, 100)]])
    assert len(merged) == 3


def test_visits_and_sweeps():
    rows = [{"frame": str(f), "region_id": r}
            for f, r in [(1, 1), (2, 1), (3, 1), (50, 2), (51, 2), (100, 1), (101, 1)]]
    v = find_visits(rows)
    assert [(x["region"], x["sweep"]) for x in v] == [("1", 1), ("2", 1), ("1", 2)]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
