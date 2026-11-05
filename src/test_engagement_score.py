"""Unit tests for engagement_score.py with synthetic data.

Run from the src folder:  python test_engagement_score.py
"""

import numpy as np

from engagement_score import CONFIG, window_score, level_of, second_scores

FPS = 25
N = 150  # 6 s


def run(yaw, pitch, ear=0.28, hand=0.0, nan_fraction=0.0):
    yaw = np.full(N, yaw, dtype=float)
    pitch = np.full(N, pitch, dtype=float)
    ear = np.full(N, ear, dtype=float) if np.isscalar(ear) else np.asarray(ear, float)
    hand = np.full(N, hand, dtype=float)
    if nan_fraction:
        k = int(N * nan_fraction)
        yaw[:k] = np.nan
        pitch[:k] = np.nan
    return window_score(yaw, pitch, ear, hand, FPS)


def test_front_is_high():
    r = run(0, 10)
    assert r["score"] == 100.0 and r["level"] == "high", r


def test_turned_away_is_low():
    r = run(70, 10)
    assert r["score"] == 0.0 and r["level"] == "low", r


def test_half_turned_is_partial():
    r = run(35, 10)   # halfway between 20 and 50 deg
    assert abs(r["score"] - 50.0) < 1.0, r


def test_head_down_uses_fixed_weight():
    r = run(60, 40)   # yaw is ignored when the head is down
    assert abs(r["score"] - 60.0) < 0.1, r


def test_low_coverage_is_unknown_not_low():
    r = run(0, 10, nan_fraction=0.6)
    assert r["score"] is None and r["level"] == "unknown", r


def test_long_closure_lowers_score():
    ear = np.full(N, 0.30)
    ear[50:90] = 0.05          # 1.6 s closed
    r = run(0, 10, ear=ear)
    assert r["long_closure"] == 1 and r["score"] < 70, r


def test_raised_hand_adds_bonus_but_caps_at_100():
    r = run(35, 10, hand=1.0)
    assert abs(r["score"] - 70.0) < 1.0, r
    assert run(0, 10, hand=1.0)["score"] == 100.0


def test_levels():
    assert level_of(CONFIG["HIGH_AT"]) == "high"
    assert level_of(CONFIG["LOW_BELOW"] - 0.1) == "low"
    assert level_of(55) == "medium"
    assert level_of(None) == "unknown"


def seconds_run(yaw, pitch, ear=0.28, n=FPS * 7):
    return second_scores(
        np.asarray(yaw, float) * np.ones(n), np.asarray(pitch, float) * np.ones(n),
        np.full(n, ear), np.zeros(n), FPS,
    )


def test_seconds_show_when_the_student_looked_away():
    n = FPS * 7
    yaw = np.zeros(n)
    yaw[FPS * 4:] = 80.0            # turns away at second 4
    rows = seconds_run(yaw, 10)
    scores = [r["score"] for r in rows]
    assert len(rows) == 7
    assert all(s == 100.0 for s in scores[:4]), scores
    assert all(s == 0.0 for s in scores[4:]), scores


def test_seconds_average_matches_window_score():
    n = FPS * 6
    yaw = np.zeros(n)
    yaw[FPS * 3:] = 35.0            # half-attentive in the second half
    rows = seconds_run(yaw, 10, n=n)
    window = window_score(yaw, np.full(n, 10.0), np.full(n, 0.28), np.zeros(n), FPS)
    mean = float(np.mean([r["score"] for r in rows]))
    assert abs(mean - window["score"]) < 1.0, (mean, window["score"])


def test_second_without_head_direction_is_unknown():
    n = FPS * 4
    yaw = np.zeros(n)
    yaw[FPS * 2: FPS * 3] = np.nan  # second 2 not measured
    rows = seconds_run(yaw, 10, n=n)
    assert rows[2]["score"] is None and rows[2]["level"] == "unknown"
    assert rows[1]["score"] == 100.0 and rows[3]["score"] == 100.0


def test_closed_eyes_lower_only_their_second():
    n = FPS * 6
    ear = np.full(n, 0.30)
    ear[FPS * 2 + 3: FPS * 3 + 15] = 0.05      # closed for about 1.5 s
    rows = second_scores(np.zeros(n), np.full(n, 10.0), ear, np.zeros(n), FPS)
    assert rows[0]["score"] == 100.0 and rows[5]["score"] == 100.0
    assert rows[2]["score"] < 80 and rows[3]["score"] < 90, [r["score"] for r in rows]


def test_short_last_slice_is_dropped():
    rows = seconds_run(0, 10, n=FPS * 3 + 5)    # 3.2 s: last slice too short
    assert len(rows) == 3


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
