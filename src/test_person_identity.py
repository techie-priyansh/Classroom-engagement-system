"""Tests for person_identity.py on synthetic data (no camera, no face model).

Run from the src folder:  python test_person_identity.py
"""

import numpy as np

from person_identity import assign_persons, read_track_faces
from visits import find_visits

rng = np.random.RandomState(7)


def unit(v):
    return v / np.linalg.norm(v)


FACES = [unit(rng.randn(512)) for _ in range(6)]       # six different people


def noisy(face, noise=0.15):
    return unit(face + noise * unit(rng.randn(512)))


def make(visit_defs):
    """
    visit_defs: list of (first_frame, region, [(track, person, seat, x)]).
    Returns rows, seat_of_row, faces keyed by (visit index, track).
    """

    rows, seats, faces = [], [], {}

    for v, (first, region, members) in enumerate(visit_defs):

        for track, person, seat, x in members:

            for f in range(first, first + 60):
                rows.append({"frame": str(f), "region_id": str(region),
                             "student_id": str(track),
                             "head_x": str(x), "head_y": "300",
                             "yaw": "0", "pitch": "10"})
                seats.append(seat)

            if person is not None:
                faces[(v, str(track))] = [noisy(FACES[person]) for _ in range(5)]

    return rows, seats, faces


def run(defs):
    rows, seats, faces = make(defs)
    visits = find_visits(rows)
    person_of_row, tracks, persons = assign_persons(rows, seats, visits, faces)
    return rows, person_of_row, persons


def person_of(rows, person_of_row, track, frame_min):
    got = {p for r, p in zip(rows, person_of_row)
           if r["student_id"] == str(track) and int(r["frame"]) >= frame_min}
    assert len(got) == 1, got
    return got.pop()


def test_same_faces_and_seats_link_across_visits():
    rows, p, persons = run([
        (1, 1, [(1, 0, "R1-S1", 100), (2, 1, "R1-S2", 300)]),
        (200, 1, [(11, 0, "R1-S1", 100), (12, 1, "R1-S2", 300)]),
    ])
    assert len(persons) == 2
    assert person_of(rows, p, 1, 1) == person_of(rows, p, 11, 200)
    assert person_of(rows, p, 2, 1) == person_of(rows, p, 12, 200)
    assert person_of(rows, p, 1, 1) != person_of(rows, p, 2, 1)


def test_student_who_changes_seat_is_followed_by_face():
    rows, p, persons = run([
        (1, 1, [(1, 0, "R1-S1", 100), (2, 1, "R1-S2", 300)]),
        (200, 1, [(11, 0, "R1-S5", 900), (12, 1, "R1-S2", 300)]),    # person 0 moved
    ])
    assert person_of(rows, p, 1, 1) == person_of(rows, p, 11, 200)


def test_two_students_swapping_seats_are_kept_apart():
    rows, p, persons = run([
        (1, 1, [(1, 0, "R1-S1", 100), (2, 1, "R1-S2", 300)]),
        (200, 1, [(11, 1, "R1-S1", 100), (12, 0, "R1-S2", 300)]),    # swapped
    ])
    assert person_of(rows, p, 1, 1) == person_of(rows, p, 12, 200)
    assert person_of(rows, p, 2, 1) == person_of(rows, p, 11, 200)


def test_new_student_on_an_old_seat_is_a_new_person():
    rows, p, persons = run([
        (1, 1, [(1, 0, "R1-S1", 100)]),
        (200, 1, [(11, 3, "R1-S1", 100)]),                          # other face, same seat
        (400, 1, [(21, 3, "R1-S1", 100)]),
    ])
    assert person_of(rows, p, 1, 1) != person_of(rows, p, 11, 200)
    assert person_of(rows, p, 11, 200) == person_of(rows, p, 21, 400)


def test_track_without_face_is_linked_by_seat():
    rows, p, persons = run([
        (1, 1, [(1, 0, "R1-S1", 100)]),
        (200, 1, [(11, None, "R1-S1", 100)]),                       # head down, no face
    ])
    assert person_of(rows, p, 1, 1) == person_of(rows, p, 11, 200)


def test_track_without_face_and_seat_gets_no_person():
    rows, p, persons = run([
        (1, 1, [(1, 0, "R1-S1", 100)]),
        (200, 1, [(11, 0, "R1-S1", 100), (12, None, "", 700)]),
        (400, 1, [(21, 0, "R1-S1", 100)]),
    ])
    got = {pp for r, pp in zip(rows, p) if r["student_id"] == "12"}
    assert got == {""}, got


def test_students_visible_together_are_never_merged():
    # same face on two simultaneous tracks (e.g. a duplicate detection)
    rows, p, persons = run([
        (1, 1, [(1, 0, "R1-S1", 100), (2, 0, "R1-S2", 300)]),
        (200, 1, [(11, 0, "R1-S1", 100)]),
    ])
    assert person_of(rows, p, 1, 1) != person_of(rows, p, 2, 1)


def test_regions_are_separate():
    rows, p, persons = run([
        (1, 1, [(1, 0, "R1-S1", 100)]),
        (100, 2, [(5, 0, "R2-S1", 100)]),
        (200, 1, [(11, 0, "R1-S1", 100)]),
        (300, 2, [(15, 0, "R2-S1", 100)]),
    ])
    names = {x for x in p if x}
    assert {n.split("-")[0] for n in names} == {"R1", "R2"}
    assert person_of(rows, p, 1, 1) == person_of(rows, p, 11, 200)


def test_passer_by_seen_once_is_dropped():
    rows, p, persons = run([
        (1, 1, [(1, 0, "R1-S1", 100), (2, 4, "", 500)]),
        (200, 1, [(11, 0, "R1-S1", 100)]),
        (400, 1, [(21, 0, "R1-S1", 100)]),
    ])
    got = {pp for r, pp in zip(rows, p) if r["student_id"] == "2" and int(r["frame"]) < 100}
    assert got == {""}, got


class FakeReader:
    """Returns one face per crop, with a fixed embedding and width."""

    def __init__(self, width):
        self.width = width

    def read(self, crop):
        return [{"w": self.width, "score": 0.9, "cx": 112.0, "cy": 112.0,
                 "emb": FACES[0]}]


def test_read_track_faces_filters_small_and_turned_faces():
    rows = []
    for f in range(1, 41):
        rows.append({"frame": str(f), "region_id": "1", "student_id": "1",
                     "head_x": "500", "head_y": "400",
                     "yaw": "0" if f <= 20 else "70", "pitch": "10"})

    visits = find_visits(rows)
    image = np.zeros((1080, 1920, 3), np.uint8)

    got = read_track_faces(rows, visits, lambda n: image, FakeReader(60), every=5)
    assert len(got[(0, "1")]) == 4          # frames 1, 6, 11, 16 (the rest are turned)

    got = read_track_faces(rows, visits, lambda n: image, FakeReader(30), every=5)
    assert got == {}                         # too small


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
