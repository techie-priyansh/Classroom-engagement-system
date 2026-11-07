"""
Combined identity: seat position + face recognition.

Seat identity alone fails when a student changes seat, and face recognition
alone fails when the face is turned away or too small. Here both are used:

  1. seat_identity.py gives every track a seat (position evidence).
  2. A face embedding is computed for every track from frames where the face
     is large enough and roughly frontal (face evidence). The embedding is the
     mean over the good frames of the track in that visit.
  3. Tracks are grouped into PERSONS by average-linkage clustering inside a
     region:
       * two tracks with face embeddings are the same person when the cosine
         similarity is high (>= FACE_MERGE after the seat correction below);
         the score gets +SEAT_BONUS when the seats agree and -SEAT_PENALTY
         when they differ, so a student who moved is still linked by face,
         but a doubtful face match on a different seat is refused;
       * a track without a usable face is linked by seat alone (seat-only
         fallback), and only when no seat conflict exists;
       * two tracks that are visible at the same time in one visit are never
         the same person (cannot-link).
  4. A track that gets no link and no seat/face stays without a person.

Output: the input CSV plus a `person_id` column (R<region>-P<number>) and a
`persons.csv` report. `engagement_score.py` uses `person_id` when present.

Run from the repository root:

    python src/person_identity.py <video> results/video_test/student_data1.csv [out_dir] [--reuse-faces]

Needs `pip install insightface onnxruntime` (the model buffalo_l is downloaded
on first use, about 280 MB).

Limits: face matching was measured on 8 students per region (one clip); with a
much larger class the face threshold must be re-checked. Faces are read from
a small crop around each head, sampled every FACE_EVERY frames.
"""

import csv
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from seat_identity import (
    MAX_OVERLAP, MIN_TRACK_FRAMES, MIN_TRACK_SHARE, MIN_VISITS_FOR_RULE,
    assign_seats, _head,
)
from visits import find_visits, visit_index


# --------------------------------------------------
# Settings (measured on the 2-minute test clip: same person 0.86 median,
# other students in the same region 0.12 median, 99th percentile 0.31)
# --------------------------------------------------

FACE_EVERY = 10             # use every 10th settled frame of a visit for faces
FACE_MIN_WIDTH = 40         # px, narrower faces are not used
FACE_MIN_SCORE = 0.5        # detector confidence
FACE_MAX_YAW = 35.0         # degrees, from head_pose (more turned: no embedding)
FACE_MAX_PITCH = 40.0
FACE_CROP_HALF = 112        # px around the head point

MIN_FACES_PER_TRACK = 2     # frames with a good face needed for an embedding

FACE_MERGE = 0.40           # linking threshold on the (corrected) face similarity
SEAT_BONUS = 0.10           # added when the seats of the two tracks agree
SEAT_PENALTY = 0.10         # subtracted when they differ
SEAT_ONLY_SCORE = 0.45      # score of a link by seat alone (no face on one side)

MIN_PERSON_VISITS = 2       # a person seen in fewer visits is dropped ...
                            # ... when the region has MIN_VISITS_FOR_RULE visits


# --------------------------------------------------
# Face reading (needs insightface; everything else works without it)
# --------------------------------------------------

class FaceReader:
    """insightface buffalo_l, detection + recognition only, on small crops."""

    def __init__(self):
        from insightface.app import FaceAnalysis

        self.app = FaceAnalysis(
            name="buffalo_l", providers=["CPUExecutionProvider"],
            allowed_modules=["detection", "recognition"])
        self.app.prepare(ctx_id=-1, det_size=(320, 320), det_thresh=0.3)

    def read(self, crop):
        """Faces in a crop: list of dicts {w, score, cx, cy, emb}."""

        return [
            {"w": float(f.bbox[2] - f.bbox[0]), "score": float(f.det_score),
             "cx": float((f.bbox[0] + f.bbox[2]) / 2),
             "cy": float((f.bbox[1] + f.bbox[3]) / 2),
             "emb": f.normed_embedding}
            for f in self.app.get(crop)
        ]


def _good_pose(row):

    try:
        return (abs(float(row["yaw"])) <= FACE_MAX_YAW
                and abs(float(row["pitch"])) <= FACE_MAX_PITCH)
    except (KeyError, TypeError, ValueError):
        return False


def read_track_faces(rows, visits, read_frame, reader, every=FACE_EVERY):
    """
    Face embeddings per track.

    Args:
        rows: CSV rows with frame, student_id, head_x, head_y, yaw, pitch
        visits: from visits.find_visits
        read_frame: function(frame_number) -> BGR image (no caching needed)
        reader: object with read(crop) -> list of {w, score, cx, cy, emb}

    Returns:
        {(visit, student_id): [embedding, ...]} (only good faces)
    """

    visit_of = visit_index(visits)

    by_frame = defaultdict(list)

    for r in rows:
        by_frame[int(r["frame"])].append(r)

    faces = defaultdict(list)

    for v, visit in enumerate(visits):

        numbers = sorted(n for n in by_frame if visit["first"] <= n <= visit["last"])

        for n in numbers[::every]:

            wanted = [r for r in by_frame[n] if _head(r) and _good_pose(r)]

            if not wanted:
                continue

            image = read_frame(n)
            height, width = image.shape[:2]

            for r in wanted:

                x, y = _head(r)
                x0 = int(max(0, x - FACE_CROP_HALF))
                y0 = int(max(0, y - FACE_CROP_HALF))
                x1 = int(min(width, x + FACE_CROP_HALF))
                y1 = int(min(height, y + FACE_CROP_HALF))

                if x1 - x0 < 60 or y1 - y0 < 60:
                    continue

                found = [
                    f for f in reader.read(image[y0:y1, x0:x1])
                    if f["w"] >= FACE_MIN_WIDTH and f["score"] >= FACE_MIN_SCORE
                ]

                if not found:
                    continue

                # the face nearest to the head point belongs to this track
                best = min(found, key=lambda f: np.hypot(
                    f["cx"] - (x - x0), f["cy"] - (y - y0)))

                faces[(v, r["student_id"])].append(best["emb"])

    return faces


# --------------------------------------------------
# Tracks and clustering (pure functions, no files, no models)
# --------------------------------------------------

def build_tracks(rows, seat_of_row, visits, faces):
    """
    One entry per (visit, tracker ID) long enough to matter.

    Returns a list of dicts: key, visit, region, frames (set), seat ("" = none),
    emb (unit vector or None), n_faces, rows (indices into `rows`).
    """

    visit_of = visit_index(visits)

    grouped = defaultdict(list)

    for i, r in enumerate(rows):
        v = visit_of(int(r["frame"]))
        if v is not None:
            grouped[(v, r["student_id"])].append(i)

    tracks = []

    for (v, tid), indices in sorted(grouped.items()):

        visit = visits[v]
        frames = {int(rows[i]["frame"]) for i in indices}

        length = visit["last"] - visit["first"] + 1

        if len(frames) < max(MIN_TRACK_FRAMES, MIN_TRACK_SHARE * length):
            continue

        votes = Counter(seat_of_row[i] for i in indices if seat_of_row[i])
        seat = ""

        if votes:
            name, count = votes.most_common(1)[0]
            if count / len(indices) >= 0.5:
                seat = name

        embs = faces.get((v, tid), [])
        emb = None

        if len(embs) >= MIN_FACES_PER_TRACK:
            m = np.mean(embs, axis=0)
            emb = m / np.linalg.norm(m)

        tracks.append({
            "key": (v, tid), "visit": v, "region": visit["region"],
            "frames": frames, "seat": seat, "emb": emb,
            "n_faces": len(embs), "rows": indices,
        })

    return tracks


def _overlap(a, b):
    return len(a["frames"] & b["frames"]) / max(1, min(len(a["frames"]), len(b["frames"])))


def link_tracks(tracks):
    """
    Group tracks into persons (list of lists of track indices).

    Average-linkage agglomerative merging inside a region; the best allowed
    pair of groups is merged until none passes the threshold.
    """

    n = len(tracks)

    face_sum = np.zeros((n, n))
    face_cnt = np.zeros((n, n))
    agree = np.zeros((n, n))
    differ = np.zeros((n, n))
    forbid = np.zeros((n, n), dtype=bool)

    for i in range(n):
        for j in range(i + 1, n):

            a, b = tracks[i], tracks[j]

            if a["region"] != b["region"]:
                forbid[i, j] = True
                continue

            if a["visit"] == b["visit"] and _overlap(a, b) > MAX_OVERLAP:
                forbid[i, j] = True
                continue

            if a["emb"] is not None and b["emb"] is not None:
                face_sum[i, j] = float(a["emb"] @ b["emb"])
                face_cnt[i, j] = 1

            if a["seat"] and b["seat"]:
                if a["seat"] == b["seat"]:
                    agree[i, j] = 1
                else:
                    differ[i, j] = 1

    for m in (face_sum, face_cnt, agree, differ):
        m += m.T
    forbid = forbid | forbid.T

    groups = {i: [i] for i in range(n)}

    def score(p, q):
        """Linking score of two groups, or None when they must not merge."""

        if forbid[p, q]:
            return None

        if face_cnt[p, q] > 0:
            s = face_sum[p, q] / face_cnt[p, q]
            seats = agree[p, q] + differ[p, q]
            if seats:
                s += SEAT_BONUS * agree[p, q] / seats - SEAT_PENALTY * differ[p, q] / seats
            return s if s >= FACE_MERGE else None

        if agree[p, q] > 0 and differ[p, q] == 0:
            return SEAT_ONLY_SCORE

        return None

    while True:

        best, pair = None, None
        ids = sorted(groups)

        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                s = score(ids[a], ids[b])
                if s is not None and (best is None or s > best):
                    best, pair = s, (ids[a], ids[b])

        if pair is None:
            break

        p, q = pair

        for m in (face_sum, face_cnt, agree, differ):
            m[p, :] += m[q, :]
            m[:, p] += m[:, q]
            m[p, p] = 0
        forbid[p, :] |= forbid[q, :]
        forbid[:, p] |= forbid[:, q]
        forbid[p, p] = False

        groups[p] += groups.pop(q)

    return list(groups.values())


def assign_persons(rows, seat_of_row, visits, faces):
    """
    Returns (person per row ("" = none), tracks, groups).

    Person names are R<region>-P<n>, numbered left to right by the median
    head position of the person.
    """

    tracks = build_tracks(rows, seat_of_row, visits, faces)
    groups = link_tracks(tracks)

    region_visits = Counter(v["region"] for v in visits)

    kept = []

    for g in groups:

        region = tracks[g[0]]["region"]
        seen = {tracks[t]["visit"] for t in g}

        # somebody seen in one visit only is most likely passing by
        if region_visits[region] >= MIN_VISITS_FOR_RULE and len(seen) < MIN_PERSON_VISITS:
            continue

        xs = [_head(rows[i])[0] for t in g for i in tracks[t]["rows"] if _head(rows[i])]

        kept.append((region, float(np.median(xs)) if xs else 0.0, g))

    kept.sort(key=lambda k: (k[0], k[1]))

    person_of_row = [""] * len(rows)
    counter = Counter()

    persons = []

    for region, _, g in kept:

        counter[region] += 1
        name = f"R{region}-P{counter[region]}"
        persons.append({"person": name, "region": region, "tracks": g})

        for t in g:
            for i in tracks[t]["rows"]:
                person_of_row[i] = name

    return person_of_row, tracks, persons


def seat_face_report(tracks, persons):
    """Where seat and face evidence disagree (for the printed summary)."""

    moved = 0
    shared = 0

    seat_people = defaultdict(set)

    for p in persons:

        seats = Counter(tracks[t]["seat"] for t in p["tracks"] if tracks[t]["seat"])

        if len(seats) > 1:
            moved += 1

        for s in seats:
            seat_people[s].add(p["person"])

    shared = sum(1 for people in seat_people.values() if len(people) > 1)

    return {"persons_on_several_seats": moved, "seats_with_several_persons": shared}


# --------------------------------------------------
# Files
# --------------------------------------------------

def draw_person_checks(rows, person_of_row, visits, read_frame, out_dir):
    """One image per region: middle frame of every sweep with person labels."""

    visit_of = visit_index(visits)

    heads = defaultdict(list)

    for r, person in zip(rows, person_of_row):

        v = visit_of(int(r["frame"]))

        if v is None or not _head(r):
            continue

        if int(r["frame"]) == (visits[v]["first"] + visits[v]["last"]) // 2:
            heads[v].append((*_head(r), person))

    paths = []

    for region in sorted({v["region"] for v in visits}):

        tiles = []

        for v, visit in enumerate(visits):

            if visit["region"] != region:
                continue

            image = read_frame((visit["first"] + visit["last"]) // 2).copy()

            for x, y, person in heads[v]:

                colour = (0, 255, 255) if person else (0, 0, 255)
                cv2.circle(image, (int(x), int(y)), 7, colour, -1)

                if person:
                    cv2.putText(image, person.split("-")[1], (int(x) - 25, int(y) - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 3)

            text = f"Region {region}  sweep {visit['sweep']}"
            cv2.putText(image, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 6)
            cv2.putText(image, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 2)

            tiles.append(cv2.resize(image, (960, 540)))

        while len(tiles) % 2:
            tiles.append(np.zeros_like(tiles[0]))

        grid = np.vstack([np.hstack(tiles[k:k + 2]) for k in range(0, len(tiles), 2)])

        path = Path(out_dir) / f"person_check_R{region}.jpg"
        cv2.imwrite(str(path), grid, [cv2.IMWRITE_JPEG_QUALITY, 88])
        paths.append(path)

    return paths


def main():

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    reuse = "--reuse-faces" in sys.argv

    if len(args) < 2:
        print(__doc__)
        return 1

    video, csv_path = Path(args[0]), Path(args[1])
    out_dir = Path(args[2]) if len(args) > 2 else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    if "head_x" not in fieldnames:
        raise SystemExit("The CSV has no head_x / head_y columns. Run video_pipeline.py again.")

    cap = cv2.VideoCapture(str(video))

    if not cap.isOpened():
        raise SystemExit(f"Unable to open video: {video}")

    def read_frame(number):
        cap.set(cv2.CAP_PROP_POS_FRAMES, number - 1)         # frames are 1-based
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"No frame {number} in {video}")
        return frame

    cache = {}

    def cached_frame(number):
        if number not in cache:
            cache[number] = read_frame(number)
        return cache[number]

    seat_of_row, _, _ = assign_seats(rows, cached_frame)
    cache.clear()

    visits = find_visits(rows)

    faces_file = out_dir / "track_faces.pkl"

    if reuse and faces_file.exists():
        faces = pickle.loads(faces_file.read_bytes())
        print(f"face embeddings read from {faces_file}")
    else:
        print("reading faces (slow on CPU: one small crop per head per sampled frame) ...")
        faces = dict(read_track_faces(rows, visits, read_frame, FaceReader()))
        faces_file.write_bytes(pickle.dumps(faces))

    person_of_row, tracks, persons = assign_persons(rows, seat_of_row, visits, faces)

    for r, seat, person in zip(rows, seat_of_row, person_of_row):
        r["seat_id"] = seat
        r["person_id"] = person

    out_csv = out_dir / f"{csv_path.stem}_ids.csv"

    with open(out_csv, "w", newline="") as f:
        columns = [c for c in fieldnames if c not in ("seat_id", "person_id")]
        writer = csv.DictWriter(f, fieldnames=columns + ["seat_id", "person_id"])
        writer.writeheader()
        writer.writerows(rows)

    with open(out_dir / "persons.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["person", "region", "visits", "tracks", "tracks_with_face",
                         "seats"])
        for p in persons:
            ts = [tracks[t] for t in p["tracks"]]
            writer.writerow([
                p["person"], p["region"], len({t["visit"] for t in ts}), len(ts),
                sum(1 for t in ts if t["emb"] is not None),
                " ".join(sorted({t["seat"] for t in ts if t["seat"]})),
            ])

    images = draw_person_checks(rows, person_of_row, visits, read_frame, out_dir)

    cap.release()

    assigned = sum(1 for p in person_of_row if p)
    with_face = sum(1 for t in tracks if t["emb"] is not None)
    with_seat = sum(1 for t in tracks if t["seat"])

    print(f"{len(rows)} rows, {assigned} with a person ({assigned / len(rows):.1%}) -> {out_csv}")
    print(f"{len(tracks)} tracks: {with_face} with a face embedding, {with_seat} with a seat")

    for region in sorted({p["region"] for p in persons}):
        print(f"  Region {region}: {sum(1 for p in persons if p['region'] == region)} persons")

    report = seat_face_report(tracks, persons)
    print(f"seat vs face: {report['persons_on_several_seats']} persons on several seats, "
          f"{report['seats_with_several_persons']} seats with several persons")

    for p in images:
        print("check image:", p)

    return 0


if __name__ == "__main__":
    sys.exit(main())
