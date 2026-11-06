"""
Engagement score per student and per settled camera visit (about 8 s).

Input : the pipeline CSV (student_data1.csv)
Output: student_scores.csv           one row per student per visit
        interval_scores.csv          class average per visit
        student_seconds.csv          one row per student per SECOND
        class_seconds.csv            class average per second
        student_timeline.png         one dot per student per visit
        student_seconds_timeline.png per-second lines + class mean, IDs on the lines
        student_seconds_grid.png     one small panel per student (short clips only)

The per-second score uses the same rules on 1-second slices. The eye
baseline comes from the whole visit (one second is too short to know a
student's normal EAR); a closure longer than 0.5 s marks every second it
touches.

HOW THE SCORE IS BUILT (all numbers are in CONFIG below, nothing is hidden)

  1. Attention, per frame, from the head direction:
       looking at the front   -> 1.0
       turning away           -> falls linearly from 1.0 (yaw 20 deg) to 0 (yaw 50 deg)
       looking down           -> DOWN_WEIGHT (0.6): writing / reading is on-task,
                                 but a head-down student cannot be told apart
                                 from a bored one with this camera
  2. Eyes (only when the eyes could be judged, see eye_features.py):
       PERCLOS and long closures reduce the score.
  3. Raised hand adds a bonus.
  4. The window (or 1-second) score is the mean over the frames where the
     head direction was measured, scaled to 0-100.
  5. If the head direction was measured in fewer than MIN_COVERAGE of the
     frames, the score is NOT given (None): "unknown" is not "low".

IMPORTANT ASSUMPTIONS (first version, not calibrated against labels)
  * "Front" is yaw FRONT_YAW and pitch FRONT_PITCH as the camera sees the
    head. This is right only if the camera is next to the board / teacher,
    so that facing the camera means facing the front. Checked against the
    30 s test clip: median yaw -11, median pitch +10.
  * With room measurements, replace this by a per-seat line of sight
    (seat position -> board / teacher / projector).
  * The weights and the High / Medium / Low limits are starting values.
  * Student IDs restart at each visit until the seat map is used, so the
    score of a student can be compared across visits only after seat-based
    identity is added (column `student_id` can then be replaced by a seat id).

Run from the repository root:

    python src/engagement_score.py results/video_test/student_data1.csv
"""

from paths import RESULTS_DIR
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from eye_features import (
    LONG_CLOSURE_SECONDS, closed_frames, closed_runs, eye_features,
)


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

CONFIG = {
    # where "looking at the front" is, as seen by the camera (degrees)
    "FRONT_YAW": 0.0,
    "FRONT_PITCH": 10.0,

    "YAW_FULL_DEG": 20.0,       # within this: fully attentive
    "YAW_ZERO_DEG": 50.0,       # beyond this: attention 0

    "DOWN_PITCH_DEG": 15.0,     # head lowered this much below FRONT_PITCH = "down"
    "DOWN_WEIGHT": 0.6,         # attention given to a lowered head

    "PERCLOS_FULL": 0.30,       # PERCLOS at which the eye penalty is maximal
    "EYE_MAX_PENALTY": 0.30,    # share of the score removed at that PERCLOS
    "LONG_CLOSURE_PENALTY": 0.20,

    "HAND_BONUS": 0.20,         # added when the hand is up for the whole window

    "MIN_COVERAGE": 0.50,       # share of frames with a measured head direction

    "HIGH_AT": 70.0,            # score >= this: High
    "LOW_BELOW": 40.0,          # score < this: Low
}


# --------------------------------------------------
# Core functions (no file access, easy to test)
# --------------------------------------------------

def frame_attention(yaw, pitch, cfg=CONFIG):
    """
    Attention (0-1) for every frame; NaN where the head direction is missing.

    yaw, pitch: arrays in degrees
    """

    yaw = np.asarray(yaw, dtype=np.float64)
    pitch = np.asarray(pitch, dtype=np.float64)

    away = np.abs(yaw - cfg["FRONT_YAW"])

    span = cfg["YAW_ZERO_DEG"] - cfg["YAW_FULL_DEG"]
    from_yaw = np.clip(1.0 - (away - cfg["YAW_FULL_DEG"]) / span, 0.0, 1.0)

    head_down = (pitch - cfg["FRONT_PITCH"]) > cfg["DOWN_PITCH_DEG"]

    # A lowered head: the yaw is unreliable (the face is seen from above),
    # so a fixed weight is used instead of the yaw rule.
    attention = np.where(head_down, cfg["DOWN_WEIGHT"], from_yaw)

    return np.where(np.isnan(yaw) | np.isnan(pitch), np.nan, attention)


def eye_penalty(perclos, long_closure, cfg=CONFIG):
    """Share of the score removed because of closed eyes (0 - about 0.5)."""

    return (
        cfg["EYE_MAX_PENALTY"] * min(1.0, perclos / cfg["PERCLOS_FULL"])
        + cfg["LONG_CLOSURE_PENALTY"] * long_closure
    )


def window_score(yaw, pitch, ear, hand_raised, fps, cfg=CONFIG):
    """
    Score of ONE student in ONE visit.

    Returns a dict with the score (0-100 or None), the level and the parts
    it was built from, so every number can be explained.
    """

    yaw = np.asarray(yaw, dtype=np.float64)
    pitch = np.asarray(pitch, dtype=np.float64)

    frames = len(yaw)

    attention = frame_attention(yaw, pitch, cfg)
    measured = ~np.isnan(attention)

    coverage = float(measured.mean()) if frames else 0.0

    head_down_share = float(
        np.mean((pitch[measured] - cfg["FRONT_PITCH"]) > cfg["DOWN_PITCH_DEG"])
    ) if measured.any() else None

    out = {
        "frames": frames,
        "coverage": round(coverage, 3),
        "attention": None,
        "head_down_share": None if head_down_share is None else round(head_down_share, 3),
        "perclos": None,
        "long_closure": None,
        "eyes_status": "unknown",
        "eye_penalty": 0.0,
        "hand_bonus": 0.0,
        "score": None,
        "level": "unknown",
    }

    if coverage < cfg["MIN_COVERAGE"] or not measured.any():
        return out

    mean_attention = float(np.mean(attention[measured]))

    # ---- eyes ----
    eyes = eye_features(ear, pitch, yaw, fps)

    out["eyes_status"] = eyes["eyes_status"]

    penalty = 0.0

    if eyes["eyes_status"] == "ok":

        out["perclos"] = eyes["perclos"]
        out["long_closure"] = eyes["long_closure"]

        penalty = eye_penalty(eyes["perclos"], eyes["long_closure"], cfg)

    # ---- hand ----
    hand = np.asarray(hand_raised, dtype=np.float64)
    bonus = cfg["HAND_BONUS"] * float(np.nanmean(hand)) if np.any(~np.isnan(hand)) else 0.0

    score = float(np.clip(mean_attention * (1.0 - penalty) + bonus, 0.0, 1.0)) * 100.0

    out.update({
        "attention": round(mean_attention, 3),
        "eye_penalty": round(penalty, 3),
        "hand_bonus": round(bonus, 3),
        "score": round(score, 1),
        "level": level_of(score, cfg),
    })

    return out


def second_scores(yaw, pitch, ear, hand_raised, fps, cfg=CONFIG):
    """
    The same score for every 1-second slice of ONE student's visit.

    The eye baseline is taken from the WHOLE visit (one second is too short
    to know a student's normal EAR); closure, PERCLOS and the long-closure
    flag are then read per second. A closure that lasts longer than
    LONG_CLOSURE_SECONDS marks every second it touches.

    A last slice shorter than half a second is dropped.

    Returns a list of dicts (second, frames, coverage, attention, perclos,
    eye_penalty, hand_bonus, score, level); score is None when the head
    direction was measured in fewer than MIN_COVERAGE of the slice's frames.
    """

    yaw = np.asarray(yaw, dtype=np.float64)
    pitch = np.asarray(pitch, dtype=np.float64)
    ear = np.asarray(ear, dtype=np.float64)
    hand = np.asarray(hand_raised, dtype=np.float64)

    attention = frame_attention(yaw, pitch, cfg)

    usable, closed, baseline = closed_frames(ear, pitch, yaw)

    long_closure_frames = np.zeros(len(yaw), dtype=bool)

    for a, b in closed_runs(closed):
        if (b - a + 1) / fps > LONG_CLOSURE_SECONDS:
            long_closure_frames[a:b + 1] = True

    size = max(1, int(round(fps)))
    out = []

    for k, start in enumerate(range(0, len(yaw), size)):

        sl = slice(start, min(len(yaw), start + size))
        frames = sl.stop - sl.start

        if frames < size / 2:
            break

        measured = ~np.isnan(attention[sl])
        coverage = float(measured.mean())

        row = {
            "second": k,
            "frames": frames,
            "coverage": round(coverage, 3),
            "attention": None,
            "perclos": None,
            "eye_penalty": 0.0,
            "hand_bonus": 0.0,
            "score": None,
            "level": "unknown",
        }

        if coverage >= cfg["MIN_COVERAGE"]:

            att = float(np.mean(attention[sl][measured]))

            penalty = 0.0

            if baseline is not None and usable[sl].mean() >= cfg["MIN_COVERAGE"]:
                perclos = float(closed[sl][usable[sl]].mean())
                row["perclos"] = round(perclos, 3)
                penalty = eye_penalty(
                    perclos, int(long_closure_frames[sl].any()), cfg
                )

            h = hand[sl]
            bonus = (
                cfg["HAND_BONUS"] * float(np.nanmean(h))
                if np.any(~np.isnan(h)) else 0.0
            )

            score = float(np.clip(att * (1.0 - penalty) + bonus, 0.0, 1.0)) * 100.0

            row.update({
                "attention": round(att, 3),
                "eye_penalty": round(penalty, 3),
                "hand_bonus": round(bonus, 3),
                "score": round(score, 1),
                "level": level_of(score, cfg),
            })

        out.append(row)

    return out


def level_of(score, cfg=CONFIG):

    if score is None:
        return "unknown"

    if score >= cfg["HIGH_AT"]:
        return "high"

    if score < cfg["LOW_BELOW"]:
        return "low"

    return "medium"


SPLIT_GAP_FRAMES = 5    # a gap in the frame numbers this long starts a new visit


# --------------------------------------------------
# CSV processing
# --------------------------------------------------

def _float(value):
    return float(value) if value not in ("", None) else np.nan


def estimate_fps(rows):

    times = sorted({(int(r["frame"]), float(r["time"])) for r in rows})

    deltas = [
        (t2 - t1) / (f2 - f1)
        for (f1, t1), (f2, t2) in zip(times, times[1:])
        if f2 - f1 == 1 and t2 > t1
    ]

    return 1.0 / float(np.median(deltas)) if deltas else 25.0


def score_csv(csv_path, cfg=CONFIG):
    """Returns (per-student visit dicts, per-student per-second dicts, fps)."""

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    if "hand_raised" not in rows[0]:
        raise SystemExit(
            "The CSV has no posture columns (hand_raised ...). "
            "Run the pipeline from the posture-cleanup commit first."
        )

    fps = estimate_fps(rows)

    # A VISIT = a block of consecutive frames (the pipeline only writes
    # settled frames, so a gap in the frame numbers is a camera move).
    frame_numbers = sorted({int(r["frame"]) for r in rows})

    visits = []
    start = prev = frame_numbers[0]

    for n in frame_numbers[1:]:
        if n - prev > SPLIT_GAP_FRAMES:
            visits.append((start, prev))
            start = n
        prev = n

    visits.append((start, prev))

    def visit_of(frame):
        for i, (a, b) in enumerate(visits):
            if a <= frame <= b:
                return i

    by_student = defaultdict(list)

    for row in rows:
        by_student[(visit_of(int(row["frame"])), row["region_id"], row["student_id"])].append(row)

    time_of = {int(r["frame"]): float(r["time"]) for r in rows}

    results = []
    seconds = []

    for (visit, region, student), student_rows in by_student.items():

        first, last = visits[visit]
        length = last - first + 1

        def column(name):
            values = np.full(length, np.nan)
            for r in student_rows:
                values[int(r["frame"]) - first] = _float(r[name])
            return values

        # Frames where the student is not in the CSV stay NaN, so a student
        # seen for only a part of the visit gets a low coverage.
        yaw, pitch = column("yaw"), column("pitch")
        ear, hand = column("avg_ear"), column("hand_raised")

        parts = window_score(yaw, pitch, ear, hand, fps, cfg)

        results.append({
            "visit": visit + 1,
            "region_id": region,
            "student_id": student,
            "start_s": time_of[first],
            "end_s": time_of[last],
            **parts,
        })

        for row in second_scores(yaw, pitch, ear, hand, fps, cfg):
            seconds.append({
                "visit": visit + 1,
                "region_id": region,
                "student_id": student,
                "time_s": round(time_of[first] + row["second"], 2),
                **row,
            })

    results.sort(key=lambda d: (d["visit"], int(d["student_id"])))
    seconds.sort(key=lambda d: (d["visit"], int(d["student_id"]), d["second"]))

    return results, seconds, fps


def interval_scores(results):
    """
    Class average per visit (all students of the same visit).
    """

    groups = defaultdict(list)

    for r in results:
        groups[(r["region_id"], r["visit"])].append(r)

    out = []

    for (region, visit), members in sorted(groups.items(), key=lambda kv: kv[0][1]):

        scored = [m["score"] for m in members if m["score"] is not None]

        out.append({
            "visit": visit,
            "region_id": region,
            "start_s": min(m["start_s"] for m in members),
            "end_s": max(m["end_s"] for m in members),
            "students": len(members),
            "students_scored": len(scored),
            "mean_score": round(float(np.mean(scored)), 1) if scored else None,
            "median_score": round(float(np.median(scored)), 1) if scored else None,
            "high": sum(m["level"] == "high" for m in members),
            "medium": sum(m["level"] == "medium" for m in members),
            "low": sum(m["level"] == "low" for m in members),
            "unknown": sum(m["level"] == "unknown" for m in members),
        })

    return out


def class_seconds(seconds):
    """Class average per second (all students of the same visit)."""

    groups = defaultdict(list)

    for r in seconds:
        groups[(r["visit"], r["region_id"], r["second"])].append(r)

    out = []

    for (visit, region, second), members in sorted(groups.items()):

        scored = [m["score"] for m in members if m["score"] is not None]

        out.append({
            "visit": visit,
            "region_id": region,
            "second": second,
            "time_s": min(m["time_s"] for m in members),
            "students": len(members),
            "students_scored": len(scored),
            "mean_score": round(float(np.mean(scored)), 1) if scored else None,
            "median_score": round(float(np.median(scored)), 1) if scored else None,
        })

    return out


def write_csv(path, rows):

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_timeline(results, path, cfg=CONFIG):
    """One line per student (id within region), time on the x axis."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.5))

    by_student = defaultdict(list)

    for r in results:
        if r["score"] is not None:
            by_student[(r["region_id"], r["student_id"])].append(
                ((r["start_s"] + r["end_s"]) / 2.0, r["score"])
            )

    for (region, student), points in by_student.items():
        t, s = zip(*sorted(points))
        ax.plot(t, s, "o-" if len(t) > 1 else "o", ms=4, lw=1, alpha=0.6)

    # student ID (as in the annotated video) next to every dot of a visit
    by_time = defaultdict(list)

    for (region, student), points in by_student.items():
        for x, y in points:
            by_time[round(x, 1)].append((student, x, y))

    for items in by_time.values():
        items.sort(key=lambda it: it[2])
        heights = spread_labels([it[2] for it in items], min_gap=3.0)
        for (student, x, y), y_label in zip(items, heights):
            ax.annotate(
                f"{student}", xy=(x, y), xytext=(x + 0.25, y_label),
                fontsize=7, va="center",
                arrowprops=dict(arrowstyle="-", lw=0.4, alpha=0.5),
            )

    ax.axhspan(cfg["HIGH_AT"], 100, color="tab:green", alpha=0.08)
    ax.axhspan(0, cfg["LOW_BELOW"], color="tab:red", alpha=0.08)

    ax.set_xlabel("time (s)")
    ax.set_ylabel("engagement score (0-100)")
    ax.set_ylim(0, 100)
    ax.set_title("Engagement per student and visit (provisional, uncalibrated)")

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def spread_labels(ys, min_gap=3.5, low=0.0, high=100.0):
    """Move label heights apart so that texts do not overlap (keeps the order)."""

    order = sorted(range(len(ys)), key=lambda i: ys[i])
    placed = {}
    previous = low - min_gap

    for i in order:
        y = max(ys[i], previous + min_gap)
        placed[i] = y
        previous = y

    # if the top ran past the limit, shift the stack back down
    overflow = previous - high

    if overflow > 0:
        for i in placed:
            placed[i] -= overflow

    return [placed[i] for i in range(len(ys))]


def _spread_labels(ys, gap, low=2.0, high=98.0):
    """Move label heights apart so that no two are closer than `gap`."""

    if not ys:
        return []

    order = sorted(range(len(ys)), key=lambda i: ys[i])
    placed = [min(max(y, low), high) for y in ys]

    for k in range(1, len(order)):
        i, j = order[k - 1], order[k]
        if placed[j] - placed[i] < gap:
            placed[j] = placed[i] + gap

    overflow = max(placed) - high
    if overflow > 0:                      # stack ran past the top: shift down
        placed = [y - overflow for y in placed]
        for k in range(len(order) - 2, -1, -1):   # and re-separate from the top
            i, j = order[k], order[k + 1]
            if placed[j] - placed[i] < gap:
                placed[i] = placed[j] - gap

    return placed


def plot_seconds(seconds, class_rows, path, cfg=CONFIG):
    """
    Per-second scores, one panel per visit. Every student line carries the
    student ID (the same number as in the annotated video and the CSV), at
    both ends, so a line can be checked against the video.
    """

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    visits = sorted({r["visit"] for r in seconds})

    lengths = [
        max(r["second"] for r in seconds if r["visit"] == v) + 1 for v in visits
    ]

    fig, axes = plt.subplots(
        1, len(visits), sharey=True, squeeze=False,
        figsize=(max(11, 4.2 * len(visits)), 6),
        gridspec_kw={"width_ratios": lengths, "wspace": 0.12},
    )

    palette = plt.get_cmap("tab20").colors

    for ax, visit, length in zip(axes[0], visits, lengths):

        rows = [r for r in seconds if r["visit"] == visit]
        region = rows[0]["region_id"]
        ids = sorted({r["student_id"] for r in rows}, key=int)

        ends = []   # (student id, colour, first point, last point)

        for n, student in enumerate(ids):

            colour = palette[n % len(palette)]

            y = np.full(length, np.nan)

            for r in rows:
                if r["student_id"] == student and r["score"] is not None:
                    y[r["second"]] = r["score"]

            if np.all(np.isnan(y)):
                continue

            x = np.arange(length) + 0.5

            ax.plot(x, y, "-o", color=colour, lw=1.3, ms=3, alpha=0.85)

            valid = np.where(~np.isnan(y))[0]
            ends.append((student, colour, valid[0], y[valid[0]], valid[-1], y[valid[-1]]))

        # IDs in the right margin, joined to the end of their line, so the
        # label stays readable even when several lines overlap (e.g. all at 100)
        label_y = _spread_labels([e[5] for e in ends], gap=4.2)

        for (student, colour, i0, y0, i1, y1), ly in zip(ends, label_y):
            ax.annotate(
                student, xy=(i1 + 0.5, y1), xytext=(length + 0.95, ly),
                color=colour, fontsize=9, fontweight="bold",
                va="center", ha="left", annotation_clip=False,
                arrowprops=dict(arrowstyle="-", color=colour, lw=0.7, alpha=0.7,
                                shrinkA=0, shrinkB=1),
            )

        mean = np.full(length, np.nan)

        for r in class_rows:
            if r["visit"] == visit and r["mean_score"] is not None:
                mean[r["second"]] = r["mean_score"]

        ax.plot(np.arange(length) + 0.5, mean, "k--", lw=2.2, alpha=0.8,
                label="class mean")

        ax.axhspan(cfg["HIGH_AT"], 100, color="tab:green", alpha=0.07)
        ax.axhspan(0, cfg["LOW_BELOW"], color="tab:red", alpha=0.07)

        t0 = rows[0]["time_s"] - rows[0]["second"]

        ax.set_xlim(-0.3, length + 1.5)
        ax.set_ylim(-2, 102)
        ax.set_xticks(np.arange(length + 1))
        ax.set_xticklabels([f"{t0 + k:.0f}" for k in range(length + 1)], fontsize=8)
        ax.set_xlabel("video time (s)")
        ax.set_title(f"Visit {visit} (Region {region})", fontsize=10)
        ax.grid(axis="y", alpha=0.25)

    axes[0][0].set_ylabel("engagement score (0-100)")
    axes[0][0].legend(loc="lower left", fontsize=8)

    fig.suptitle(
        "Engagement per second; the number at the end of each line is the "
        "student ID (dashed = class mean, uncalibrated)", fontsize=11,
    )

    fig.subplots_adjust(left=0.06, right=0.985, top=0.90, bottom=0.10)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_student_grid(seconds, path, cfg=CONFIG, columns=6, max_visits=6):
    """
    One small panel per student (title = student ID), one row of panels per
    visit. Overlapping lines cannot hide each other here, so every ID can be
    checked against the video. Skipped for long recordings (many visits).
    """

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    visits = sorted({r["visit"] for r in seconds})

    if not visits or len(visits) > max_visits:
        return False

    per_visit = {
        v: sorted({r["student_id"] for r in seconds if r["visit"] == v}, key=int)
        for v in visits
    }

    rows_needed = sum(-(-len(ids) // columns) for ids in per_visit.values())

    fig, axes = plt.subplots(
        rows_needed, columns, sharey=True, squeeze=False,
        figsize=(2.3 * columns, 1.9 * rows_needed + 0.6),
    )

    row = 0

    for visit in visits:

        ids = per_visit[visit]
        region = next(r["region_id"] for r in seconds if r["visit"] == visit)

        for n, student in enumerate(ids):

            ax = axes[row + n // columns][n % columns]

            mine = sorted(
                (r for r in seconds if r["visit"] == visit and r["student_id"] == student),
                key=lambda r: r["second"],
            )

            x = [r["second"] + 0.5 for r in mine]
            y = [np.nan if r["score"] is None else r["score"] for r in mine]

            ax.plot(x, y, "-o", ms=3, lw=1.3, color="tab:blue")
            ax.axhspan(cfg["HIGH_AT"], 100, color="tab:green", alpha=0.08)
            ax.axhspan(0, cfg["LOW_BELOW"], color="tab:red", alpha=0.08)
            ax.set_ylim(-2, 102)
            ax.set_xlim(0, max(x) + 0.5 if x else 1)
            ax.set_title(f"ID {student}  (V{visit}/R{region})", fontsize=9, fontweight="bold")
            ax.tick_params(labelsize=7)

            if all(np.isnan(y)):
                ax.text(0.5, 0.5, "unknown", transform=ax.transAxes,
                        ha="center", va="center", color="0.4", fontsize=8)

        for n in range(len(ids), -(-len(ids) // columns) * columns):
            axes[row + n // columns][n % columns].axis("off")

        row += -(-len(ids) // columns)

    fig.suptitle("Engagement per second, one panel per student "
                 "(x = seconds into the visit, y = score)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=120)
    plt.close(fig)

    return True


def main():

    csv_path = Path(sys.argv[1] if len(sys.argv) > 1
                    else str(RESULTS_DIR / "video_test" / "student_data1.csv"))

    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else csv_path.parent

    out_dir.mkdir(parents=True, exist_ok=True)

    results, seconds, fps = score_csv(csv_path)
    intervals = interval_scores(results)
    per_second = class_seconds(seconds)

    write_csv(out_dir / "student_scores.csv", results)
    write_csv(out_dir / "interval_scores.csv", intervals)
    write_csv(out_dir / "student_seconds.csv", seconds)
    write_csv(out_dir / "class_seconds.csv", per_second)
    plot_timeline(results, out_dir / "student_timeline.png")
    plot_seconds(seconds, per_second, out_dir / "student_seconds_timeline.png")

    if not plot_student_grid(seconds, out_dir / "student_seconds_grid.png"):
        print("(per-student grid skipped: too many visits; use student_seconds.csv)")

    scored = [r for r in results if r["score"] is not None]

    print(f"{len(results)} student visits ({fps:.0f} fps), {len(scored)} scored, "
          f"{len(results) - len(scored)} unknown (low coverage)")

    for v in intervals:
        print(
            f"  V{v['visit']} R{v['region_id']} {v['start_s']:6.1f}-{v['end_s']:6.1f} s: "
            f"mean {v['mean_score']}  (high {v['high']}, medium {v['medium']}, "
            f"low {v['low']}, unknown {v['unknown']})"
        )

    print(f"Written to {out_dir}")
    print("Note: weights and limits are starting values, not calibrated.")


if __name__ == "__main__":
    main()
