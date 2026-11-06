"""Unit tests for utils.head_point (no camera needed).

Run from the src folder:  python test_head_point.py
"""

from utils import head_point

BOX = (100, 200, 200, 400)   # 100 x 200 person box


def kp(**points):
    return {n: {"point": p, "confidence": c} for n, (p, c) in points.items()}


def test_nose_is_preferred():
    k = kp(nose=((150, 230), 0.9), left_eye=((140, 220), 0.9), right_eye=((160, 220), 0.9))
    assert head_point(k, BOX) == (150.0, 230.0)


def test_eyes_when_nose_is_unreliable():
    k = kp(nose=((150, 230), 0.1), left_eye=((140, 220), 0.9), right_eye=((160, 224), 0.9))
    assert head_point(k, BOX) == (150.0, 222.0)


def test_one_eye_is_enough():
    k = kp(nose=((0, 0), 0.0), left_eye=((140, 220), 0.9), right_eye=((0, 0), 0.0))
    assert head_point(k, BOX) == (140.0, 220.0)


def test_box_top_when_nothing_is_reliable():
    k = kp(nose=((150, 230), 0.1))
    assert head_point(k, BOX) == (150.0, 220.0)
    assert head_point(None, BOX) == (150.0, 220.0)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
