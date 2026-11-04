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
