import cv2


def extract_head_roi(frame, bbox, head_ratio=0.45):
    """
    Extract the upper portion of a person's bounding box,
    where the head is expected to be.

    Args:
        frame: Original BGR frame
        bbox: (x1, y1, x2, y2)
        head_ratio: Fraction of height to keep

    Returns:
        roi, (rx1, ry1, rx2, ry2)
    """

    x1, y1, x2, y2 = map(int, bbox)

    h = y2 - y1
    head_h = int(h * head_ratio)

    ry2 = y1 + head_h

    roi = frame[y1:ry2, x1:x2]

    return roi, (x1, y1, x2, ry2)


def make_square_roi(frame, bbox):
    """
    Square crop around bbox. The crop is taken first and only the small
    crop is padded (replicated border) if the square leaves the frame.

    Returns:
        roi          : square image
        square_bbox  : (x1, y1, x2, y2) in ORIGINAL frame coordinates
                       (can be negative / beyond the frame size)
    """

    x1, y1, x2, y2 = map(int, bbox)

    side = max(x2 - x1, y2 - y1)

    if side <= 0:
        return None, None

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    sx1 = cx - side // 2
    sy1 = cy - side // 2
    sx2 = sx1 + side
    sy2 = sy1 + side

    frame_h, frame_w = frame.shape[:2]

    # Part of the square that lies inside the frame
    cx1 = max(0, sx1)
    cy1 = max(0, sy1)
    cx2 = min(frame_w, sx2)
    cy2 = min(frame_h, sy2)

    if cx2 <= cx1 or cy2 <= cy1:
        return None, None

    roi = frame[cy1:cy2, cx1:cx2]

    pad_left = cx1 - sx1
    pad_top = cy1 - sy1
    pad_right = sx2 - cx2
    pad_bottom = sy2 - cy2

    if pad_left or pad_top or pad_right or pad_bottom:
        roi = cv2.copyMakeBorder(
            roi,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_REPLICATE
        )

    return roi, (sx1, sy1, sx2, sy2)


def head_point(named_keypoints, bbox, min_conf=0.3):
    """
    Where the head is in the image, as one (x, y) point in pixels. Used as the
    anchor that ties a student to a seat.

    Order of preference: nose, mean of the eyes, mean of the ears, and, when
    no keypoint is reliable, the top centre of the person box.

    Args:
        named_keypoints: dict from BodyPoseEstimator.get_named_keypoints, or None
        bbox: (x1, y1, x2, y2) of the person
    """

    def reliable(name):

        item = (named_keypoints or {}).get(name)

        if item is None:
            return None

        confidence = item.get("confidence")

        if confidence is not None and confidence < min_conf:
            return None

        return float(item["point"][0]), float(item["point"][1])

    for names in (("nose",), ("left_eye", "right_eye"), ("left_ear", "right_ear")):

        points = [p for p in (reliable(n) for n in names) if p is not None]

        if points:
            return (
                sum(p[0] for p in points) / len(points),
                sum(p[1] for p in points) / len(points),
            )

    x1, y1, x2, y2 = map(float, bbox)

    return (x1 + x2) / 2.0, y1 + 0.10 * (y2 - y1)
