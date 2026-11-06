"""Split the pipeline CSV into camera VISITS.

The pipeline writes only settled frames, so a gap in the frame numbers means
the camera moved to another region. One visit = one block of consecutive
frames of one region.
"""

SPLIT_GAP_FRAMES = 5    # a gap in the frame numbers this long starts a new visit


def find_visits(rows, gap=SPLIT_GAP_FRAMES):
    """
    Args:
        rows: dicts with "frame" and "region_id" (strings or numbers)

    Returns:
        list of dicts {"first", "last", "region"} in time order; also a
        sweep number for every visit of the same region (1, 2, 3 ...).
    """

    frames = {}

    for r in rows:
        frames.setdefault(int(r["frame"]), r["region_id"])

    numbers = sorted(frames)

    blocks = []
    start = prev = numbers[0]

    for n in numbers[1:]:
        if n - prev > gap:
            blocks.append((start, prev))
            start = n
        prev = n

    blocks.append((start, prev))

    visits = []
    seen = {}

    for first, last in blocks:

        region = str(frames[first])
        seen[region] = seen.get(region, 0) + 1

        visits.append({
            "first": first,
            "last": last,
            "region": region,
            "sweep": seen[region],
        })

    return visits


def visit_index(visits):
    """Function frame_number -> index of the visit it belongs to (or None)."""

    import bisect

    starts = [v["first"] for v in visits]

    def lookup(frame):
        i = bisect.bisect_right(starts, frame) - 1
        if i >= 0 and frame <= visits[i]["last"]:
            return i
        return None

    return lookup
