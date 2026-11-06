"""
Seat-based identity: which student is which when the camera comes back.

The tracker restarts at every visit, so IDs are only valid inside one visit.
Students stay in their seats, so identity is tied to the SEAT:

  1. For every region, the first visit is the REFERENCE view.
  2. Every later visit of the region is aligned to it (ORB features on the
     static scene, seat_map.estimate_view_shift). The camera does not come
     back to exactly the same view (on the test clip Region 1 drifted by up
     to 26 px between sweeps), and this removes that shift.
  3. SEATS of a region = head positions seen in the visits (in the reference
     view), grouped by closeness. A seat that is empty in the first visit but
     occupied later is still found.
  4. Every detection is matched to the nearest free seat (one detection per
     seat per frame, Hungarian). A tracker ID takes the seat that most of its
     frames voted for, so a student whose ID changes mid-visit still gets one
     seat, and a short ghost track far from every seat gets none.

Output: the input CSV plus a `seat_id` column (R<region>-S<number>, seats
numbered top to bottom, left to right in the reference view; empty = no seat),
`seat_map.json`, `alignment.json` and check images (seat labels drawn on the
frames of every sweep: the same label must sit on the same student).

Run from the repository root:

    python src/seat_identity.py tests/clip.mp4 results/video_test/student_data1.csv

Limits (honest): a student who changes seat is a different seat; two students
swapping seats between visits are not noticed; students closer than about
50 px (back rows at 1080p) can be merged into one seat.
"""

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

from seat_map import SeatMap, estimate_view_shift, to_reference
from visits import find_visits, visit_index


# --------------------------------------------------
# Settings
# --------------------------------------------------

CLUSTER_PX = 45.0            # used by merge_seats() (centres from several visits)
SEAT_MERGE_PX = 75.0         # one student's head moves this much (leaning, writing)
MIN_TRACK_FRAMES = 12        # a track shorter than this (0.5 s) does not make a seat
MIN_TRACK_SHARE = 0.3        # ... nor one present in less than this share of its visit
MIN_TRACK_VOTE = 0.5         # share of a track's frames that must agree on a seat
MAX_OVERLAP = 0.2            # two tracks on one seat may overlap in time this much
SEAT_MATCH_PX = 110.0        # a head further than this from every free seat gets none
                             # (a student leaning on a hand sat 87 px from his seat)
MIN_SEAT_VISITS = 2          # a seat seen in fewer visits is dropped ...
MIN_VISITS_FOR_RULE = 3      # ... when the region has at least this many visits


# --------------------------------------------------
# Core (no file access)
# --------------------------------------------------

def _head(row):

    x, y = row.get("head_x"), row.get("head_y")

    if x in ("", None) or y in ("", None):
        return None

    return float(x), float(y)


def merge_seats(position_sets, cluster_px=CLUSTER_PX):
    """Seat centres found in several visits -> one list (same seat = close)."""

    points = [p for centres in position_sets for p in centres]

    if not points:
        return np.empty((0, 2))

    points = np.asarray(points, dtype=np.float64)

    if len(points) == 1:
        return points

    groups = fcluster(linkage(points, method="average"), t=cluster_px,
                      criterion="distance")

    return np.array([points[groups == g].mean(axis=0) for g in np.unique(groups)])


def cluster_seats(points, merge_px=SEAT_MERGE_PX, max_overlap=MAX_OVERLAP):
    """
    Group the head positions of all tracks (one point per track per visit)
    into seats.

    Distance alone cannot do it: one student who leans moves his head by
    more than the distance between two neighbours in the back rows. So two
    points are NEVER merged when they come from tracks of the same visit
    that were present at the same time (two people at once are two seats),
    and otherwise points are merged, closest first, up to `merge_px`.

    Args:
        points: list of dicts {"visit", "pos": (x, y), "frames": set}

    Returns:
        list of dicts {"pos": (x, y), "visits": set, "weight": int}
    """

    clusters = [
        {"members": [p], "pos": np.array(p["pos"], dtype=np.float64),
         "weight": len(p["frames"])}
        for p in points
    ]

    def conflict(a, b):

        for p in a["members"]:
            for q in b["members"]:
                if p["visit"] != q["visit"]:
                    continue
                shared = len(p["frames"] & q["frames"])
                if shared > max_overlap * min(len(p["frames"]), len(q["frames"])):
                    return True

        return False

    while True:

        best = None

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):

                d = float(np.linalg.norm(clusters[i]["pos"] - clusters[j]["pos"]))

                if d <= merge_px and (best is None or d < best[0]) \
                        and not conflict(clusters[i], clusters[j]):
                    best = (d, i, j)

        if best is None:
            break

        _, i, j = best
        a, b = clusters[i], clusters[j]

        total = a["weight"] + b["weight"]

        merged = {
            "members": a["members"] + b["members"],
            "pos": (a["pos"] * a["weight"] + b["pos"] * b["weight"]) / total,
            "weight": total,
        }

        clusters = [c for k, c in enumerate(clusters) if k not in (i, j)] + [merged]

    return [
        {"pos": tuple(c["pos"]), "visits": {m["visit"] for m in c["members"]},
         "weight": c["weight"]}
        for c in clusters
    ]


def vote_track_seats(frame_seats, min_vote=MIN_TRACK_VOTE, max_overlap=MAX_OVERLAP):
    """
    Decide one seat per tracker ID.

    Args:
        frame_seats: {track_id: {frame: seat or None}} (per-frame matching)

    Returns:
        {track_id: seat or None}
    """

    wanted = {}

    for track, frames in frame_seats.items():

        seats = [s for s in frames.values() if s is not None]

        if not seats:
            wanted[track] = None
            continue

        seat, count = Counter(seats).most_common(1)[0]

        wanted[track] = seat if count / len(seats) >= min_vote else None

    # Two tracks on one seat are fine only if they follow each other (an ID
    # that changed); tracks present at the same time cannot be the same person.
    by_seat = defaultdict(list)

    for track, seat in wanted.items():
        if seat is not None:
            by_seat[seat].append(track)

    result = dict(wanted)

    for seat, tracks in by_seat.items():

        tracks.sort(key=lambda t: -len(frame_seats[t]))

        accepted = []

        for t in tracks:

            mine = set(frame_seats[t])

            clash = any(
                len(mine & set(frame_seats[a])) > max_overlap * len(mine)
                for a in accepted
            )

            if clash:
                result[t] = None
            else:
                accepted.append(t)

    return result


def assign_seats(rows, get_frame, cluster_px=CLUSTER_PX,
                 max_distance=SEAT_MATCH_PX):
    """
    Add a seat to every row.

    Args:
        rows: CSV rows (dicts) with frame, region_id, student_id, head_x, head_y
        get_frame: function(frame_number) -> BGR image, used for the view
                   alignment (only the middle frame of every visit is read)

    Returns:
        seat per row (list, same order as rows, "" = none),
        seat maps {region: SeatMap}, per-visit info (list of dicts)
    """

    visits = find_visits(rows)
    visit_of = visit_index(visits)

    row_visit = [visit_of(int(r["frame"])) for r in rows]

    by_visit = defaultdict(list)

    for i, v in enumerate(row_visit):
        if v is not None:
            by_visit[v].append(i)

    # ---- 1. align every visit to the first visit of its region ----
    images = {}
    reference = {}
    infos = []

    for v, visit in enumerate(visits):

        images[v] = get_frame((visit["first"] + visit["last"]) // 2)

        region = visit["region"]

        info = {
            "visit": v + 1, "region": region, "sweep": visit["sweep"],
            "first_frame": visit["first"], "last_frame": visit["last"],
            "aligned": True, "dx": 0.0, "dy": 0.0, "scale": 1.0,
            "rotation_deg": 0.0, "inliers": None, "M": None,
        }

        if region not in reference:
            reference[region] = v
        else:
            shift = estimate_view_shift(images[reference[region]], images[v])
            info.update({
                "aligned": bool(shift["ok"]),
                "dx": shift["dx"], "dy": shift["dy"],
                "scale": shift["scale"], "rotation_deg": shift["rotation_deg"],
                "inliers": shift["inliers"],
                "M": shift["M"] if shift["ok"] else None,
            })

        infos.append(info)

    # ---- 2. head positions of every visit in the reference view ----
    aligned_pos = {}      # row index -> (x, y) in the reference view

    frame_lists = defaultdict(lambda: defaultdict(list))   # visit -> frame -> [pos]

    for v, indices in by_visit.items():

        M = infos[v]["M"]

        for i in indices:

            head = _head(rows[i])

            if head is None:
                continue

            p = tuple(to_reference([head], M)[0])
            aligned_pos[i] = p
            frame_lists[v][int(rows[i]["frame"])].append(p)

    # ---- 3. seats of every region: one point per track per visit ----
    seat_maps = {}

    track_frames = defaultdict(lambda: defaultdict(set))   # visit -> track -> frames
    track_pos = defaultdict(lambda: defaultdict(list))     # visit -> track -> [pos]

    for v, indices in by_visit.items():
        for i in indices:
            if i in aligned_pos:
                track = rows[i]["student_id"]
                track_frames[v][track].add(int(rows[i]["frame"]))
                track_pos[v][track].append(aligned_pos[i])

    for region in sorted({v["region"] for v in visits}):

        mine = [v for v, visit in enumerate(visits) if visit["region"] == region]

        points = [
            {"visit": v, "pos": tuple(np.median(track_pos[v][t], axis=0)),
             "frames": track_frames[v][t]}
            for v in mine for t in track_pos[v]
            if len(track_frames[v][t]) >= max(
                MIN_TRACK_FRAMES,
                MIN_TRACK_SHARE * (visits[v]["last"] - visits[v]["first"] + 1))
        ]

        clusters = cluster_seats(points)

        # A seat seen in one visit only is most likely somebody passing by
        # (in the test clip: a person walking in front of the first row).
        if len(mine) >= MIN_VISITS_FOR_RULE:
            clusters = [c for c in clusters if len(c["visits"]) >= MIN_SEAT_VISITS]

        seat_maps[region] = SeatMap([c["pos"] for c in clusters], region)

    # ---- 4. per-frame matching, then one seat per tracker ID ----
    seat_of_row = [""] * len(rows)

    for v, indices in by_visit.items():

        seat_map = seat_maps[visits[v]["region"]]

        per_frame = defaultdict(list)

        for i in indices:
            if i in aligned_pos:
                per_frame[int(rows[i]["frame"])].append(i)

        frame_seats = defaultdict(dict)     # track -> frame -> seat

        for frame, members in per_frame.items():

            seats = seat_map.assign(
                [aligned_pos[i] for i in members], None, max_distance)

            for i, seat in zip(members, seats):
                frame_seats[rows[i]["student_id"]][frame] = seat

        decided = vote_track_seats(frame_seats)

        for i in indices:
            seat = decided.get(rows[i]["student_id"])
            seat_of_row[i] = seat or ""

        infos[v]["tracks"] = len(frame_seats)
        infos[v]["tracks_with_seat"] = sum(1 for s in decided.values() if s)

        spreads = []

        for i in indices:
            if seat_of_row[i] and i in aligned_pos:
                c = seat_map.positions[seat_map.seat_ids.index(seat_of_row[i])]
                spreads.append(float(np.hypot(*(np.array(aligned_pos[i]) - c))))

        infos[v]["median_offset_px"] = round(float(np.median(spreads)), 1) if spreads else None

    return seat_of_row, seat_maps, infos


# --------------------------------------------------
# Files and check images
# --------------------------------------------------

def draw_checks(rows, seat_of_row, seat_maps, infos, get_frame, out_dir):
    """
    One image per region: the middle frame of every sweep, side by side, with
    the seat labels drawn at the seat positions (moved into each view) and a
    dot on every detected head. The same label must sit on the same student.
    """

    visits = [
        {"first": i["first_frame"], "last": i["last_frame"], "region": i["region"],
         "sweep": i["sweep"], "M": i["M"]} for i in infos
    ]

    visit_of = visit_index(
        [{"first": v["first"], "last": v["last"]} for v in visits])

    heads = defaultdict(list)       # visit -> [(x, y, seat)]

    for r, seat in zip(rows, seat_of_row):

        v = visit_of(int(r["frame"]))
        mid = (visits[v]["first"] + visits[v]["last"]) // 2 if v is not None else None

        if v is not None and int(r["frame"]) == mid and _head(r):
            x, y = _head(r)
            heads[v].append((x, y, seat))

    paths = []

    for region, seat_map in seat_maps.items():

        mine = [v for v in range(len(visits)) if visits[v]["region"] == region]
        tiles = []

        for v in mine:

            image = get_frame((visits[v]["first"] + visits[v]["last"]) // 2).copy()

            M = visits[v]["M"]

            for seat_id, (x, y) in zip(seat_map.seat_ids, seat_map.positions):

                if M is not None:
                    x, y = (np.asarray(M) @ np.array([x, y, 1.0]))[:2]

                cv2.circle(image, (int(x), int(y)), 34, (0, 255, 255), 2)
                cv2.putText(image, seat_id.split("-")[1], (int(x) - 22, int(y) - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

            for x, y, seat in heads[v]:
                colour = (0, 255, 0) if seat else (0, 0, 255)
                cv2.circle(image, (int(x), int(y)), 7, colour, -1)

            cv2.putText(image, f"Region {region}  sweep {visits[v]['sweep']}",
                        (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 6)
            cv2.putText(image, f"Region {region}  sweep {visits[v]['sweep']}",
                        (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 2)

            tiles.append(cv2.resize(image, (960, 540)))

        while len(tiles) % 2:
            tiles.append(np.zeros_like(tiles[0]))

        grid = np.vstack([np.hstack(tiles[k:k + 2]) for k in range(0, len(tiles), 2)])

        path = Path(out_dir) / f"seat_check_R{region}.jpg"
        cv2.imwrite(str(path), grid, [cv2.IMWRITE_JPEG_QUALITY, 88])
        paths.append(path)

    return paths


def main():

    if len(sys.argv) < 3:
        print(__doc__)
        return 1

    video, csv_path = Path(sys.argv[1]), Path(sys.argv[2])

    out_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else csv_path.parent

    out_dir.mkdir(parents=True, exist_ok=True)

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if "head_x" not in fieldnames:
        raise SystemExit(
            "The CSV has no head_x / head_y columns. Run video_pipeline.py "
            "again with the current code."
        )

    cap = cv2.VideoCapture(str(video))

    if not cap.isOpened():
        raise SystemExit(f"Unable to open video: {video}")

    cache = {}

    def get_frame(number):
        if number not in cache:
            cap.set(cv2.CAP_PROP_POS_FRAMES, number - 1)      # frames are 1-based
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"No frame {number} in {video}")
            cache[number] = frame
        return cache[number]

    seat_of_row, seat_maps, infos = assign_seats(rows, get_frame)

    for row, seat in zip(rows, seat_of_row):
        row["seat_id"] = seat

    out_csv = out_dir / f"{csv_path.stem}_seats.csv"

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames + ["seat_id"])
        writer.writeheader()
        writer.writerows(rows)

    (out_dir / "seat_map.json").write_text(json.dumps({
        region: [
            {"seat_id": s, "x": round(float(x), 1), "y": round(float(y), 1)}
            for s, (x, y) in zip(m.seat_ids, m.positions)
        ] for region, m in seat_maps.items()
    }, indent=2))

    (out_dir / "alignment.json").write_text(json.dumps([
        {k: (v if k != "M" else (None if v is None else np.asarray(v).round(4).tolist()))
         for k, v in info.items()} for info in infos
    ], indent=2))

    images = draw_checks(rows, seat_of_row, seat_maps, infos, get_frame, out_dir)

    cap.release()

    assigned = sum(1 for s in seat_of_row if s)

    print(f"{len(rows)} rows, {assigned} with a seat ({assigned / len(rows):.1%}) -> {out_csv}")

    for region, m in seat_maps.items():
        print(f"  Region {region}: {len(m.positions)} seats")

    print("  visit  region sweep  shift (dx, dy) px   tracks (with seat)  offset to seat")

    for i in infos:
        print(
            f"  {i['visit']:>5}  R{i['region']:<5} {i['sweep']:>5}  "
            f"({i['dx']:+6.1f}, {i['dy']:+6.1f})"
            f"{'' if i['aligned'] else '  NOT ALIGNED'}   "
            f"{i.get('tracks', 0):>3} ({i.get('tracks_with_seat', 0):>3})"
            f"           {i.get('median_offset_px')} px"
        )

    for p in images:
        print("check image:", p)

    return 0


if __name__ == "__main__":
    sys.exit(main())
