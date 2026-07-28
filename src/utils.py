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