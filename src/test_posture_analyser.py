"""Unit tests for PostureAnalyser with synthetic keypoints (no camera needed).

Run from the src folder:  python test_posture_analyser.py
"""

import numpy as np

from body_pose import BodyPoseEstimator
from posture_analyser import PostureAnalyser


def make_person(overrides=None, conf=0.9):
    """A student facing the camera, shoulders 100 px wide."""

    points = {
        "nose": (500, 300),
        "left_eye": (485, 290), "right_eye": (515, 290),
        "left_ear": (470, 295), "right_ear": (530, 295),
        "left_shoulder": (450, 400), "right_shoulder": (550, 400),
        "left_elbow": (440, 500), "right_elbow": (560, 500),
        "left_wrist": (440, 560), "right_wrist": (560, 560),
        "left_hip": (460, 600), "right_hip": (540, 600),
        "left_knee": (460, 700), "right_knee": (540, 700),
        "left_ankle": (460, 800), "right_ankle": (540, 800),
    }

    confidences = {name: conf for name in points}

    for name, value in (overrides or {}).items():
        if isinstance(value, tuple) and len(value) == 3:
            points[name] = (value[0], value[1])
            confidences[name] = value[2]
        else:
            points[name] = value

    names = BodyPoseEstimator.KEYPOINT_NAMES

    return {
        name: {
            "point": np.array(points[name], dtype=np.float64),
            "confidence": confidences[name],
        }
        for name in names
    }


def main():

    analyser = PostureAnalyser()

    # ---- upright, hands on the desk ------------------------------------
    f = analyser.analyse(make_person())
    assert abs(f["shoulder_tilt"]) < 1e-6
    assert abs(f["head_drop"] - 1.0) < 1e-6          # 100 px / 100 px
    assert abs(f["head_offset"]) < 1e-6
    assert f["hand_raised"] == 0
    assert f["hands_near_face"] == 0

    # ---- head dropped towards the desk ---------------------------------
    f = analyser.analyse(make_person({"nose": (500, 390)}))
    assert abs(f["head_drop"] - 0.1) < 1e-6

    # ---- one shoulder lower (image right lower -> positive) ------------
    f = analyser.analyse(make_person({"right_shoulder": (550, 420)}))
    assert f["shoulder_tilt"] > 5

    # ---- the model swapping left/right must not flip the sign ----------
    f = analyser.analyse(make_person({
        "left_shoulder": (550, 420), "right_shoulder": (450, 400),
    }))
    assert f["shoulder_tilt"] > 5

    # ---- raised hand ----------------------------------------------------
    f = analyser.analyse(make_person({"right_wrist": (560, 300)}))
    assert f["hand_raised"] == 1

    # ---- hand at the chin -----------------------------------------------
    f = analyser.analyse(make_person({"left_wrist": (490, 340)}))
    assert f["hands_near_face"] == 1

    # ---- low confidence wrist is ignored, not guessed -------------------
    f = analyser.analyse(make_person({
        "left_wrist": (440, 300, 0.1), "right_wrist": (560, 300, 0.1),
    }))
    assert f["hand_raised"] is None
    assert f["hands_near_face"] is None

    # ---- low confidence nose -> head features are None ------------------
    f = analyser.analyse(make_person({"nose": (500, 300, 0.1)}))
    assert f["head_drop"] is None and f["head_offset"] is None
    assert f["shoulder_tilt"] is not None

    # ---- unreliable shoulders -> no posture at all ----------------------
    assert analyser.analyse(
        make_person({"left_shoulder": (450, 400, 0.2)})
    ) is None

    # ---- hidden knees must not matter -----------------------------------
    f = analyser.analyse(make_person({
        "left_knee": (0, 0), "right_knee": (0, 0),
        "left_hip": (0, 0), "right_hip": (0, 0),
    }))
    assert f is not None and "left_knee_angle" not in f

    # ---- shoulders too close together (tiny / far away) -----------------
    assert analyser.analyse(make_person({
        "left_shoulder": (495, 400), "right_shoulder": (505, 400),
    })) is None

    print("All posture tests passed.")


if __name__ == "__main__":
    main()
