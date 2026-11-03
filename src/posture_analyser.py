import numpy as np


class PostureAnalyser:
    """
    Upper-body posture features for SEATED students seen from the front.

    Only the head, shoulders and wrists are used. Hips and knees are
    hidden behind the desks, and the pose model invents them with high
    confidence, so they are deliberately NOT used.

    Every distance is divided by the shoulder width, so the features do
    not depend on how far the student is from the camera or on the zoom.

    A feature is None when the keypoints it needs are not reliable
    (confidence below `min_conf`), never a made-up number.
    """

    # Keypoints below this confidence are ignored.
    MIN_CONF = 0.5

    # Shoulders closer together than this (pixels) are too small to trust.
    MIN_SHOULDER_WIDTH_PX = 20.0

    # A wrist counts as "raised" when it is this many shoulder-widths
    # above the same-side shoulder.
    HAND_RAISED_MARGIN = 0.5

    # A wrist counts as "near the face" when it is closer than this many
    # shoulder-widths to the nose.
    HAND_NEAR_FACE_DIST = 0.8

    def __init__(self, min_conf=MIN_CONF):
        self.min_conf = min_conf

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _point(self, keypoints, name):
        """Return the (x, y) point, or None if it is not reliable."""

        entry = keypoints.get(name)

        if entry is None:
            return None

        point = np.asarray(entry["point"], dtype=np.float64)
        confidence = entry["confidence"]

        # (0, 0) means "not detected"
        if point[0] <= 0 and point[1] <= 0:
            return None

        if confidence is not None and confidence < self.min_conf:
            return None

        return point

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def analyse(self, keypoints):
        """
        Args:
            keypoints: dict from BodyPoseEstimator.get_named_keypoints()

        Returns:
            dict with
              shoulder_tilt   degrees, -90..90. Line between the shoulders
                              versus horizontal; positive when the shoulder
                              on the right of the IMAGE is lower.
              head_drop       (shoulder-centre y - nose y) / shoulder width.
                              Large = sitting tall; small or negative =
                              head dropped towards or below the shoulders.
              head_offset     (nose x - shoulder-centre x) / shoulder width.
                              Sideways lean or head turn.
              hand_raised     1 / 0, or None if no wrist is reliable.
              hands_near_face 1 / 0, or None if no wrist or nose is reliable.
            or None if the shoulders are not reliable.
        """

        if keypoints is None:
            return None

        left_shoulder = self._point(keypoints, "left_shoulder")
        right_shoulder = self._point(keypoints, "right_shoulder")

        if left_shoulder is None or right_shoulder is None:
            return None

        shoulder_vector = right_shoulder - left_shoulder
        shoulder_width = float(np.linalg.norm(shoulder_vector))

        if shoulder_width < self.MIN_SHOULDER_WIDTH_PX:
            return None

        shoulder_center = (left_shoulder + right_shoulder) / 2.0

        # ---- Shoulder tilt ----------------------------------------------
        # Order the two shoulders left-to-right in the image so the angle
        # does not depend on which one the model calls "left".

        if left_shoulder[0] <= right_shoulder[0]:
            image_left, image_right = left_shoulder, right_shoulder
        else:
            image_left, image_right = right_shoulder, left_shoulder

        dx = image_right[0] - image_left[0]
        dy = image_right[1] - image_left[1]

        shoulder_tilt = float(np.degrees(np.arctan2(dy, dx)))

        # ---- Head position relative to the shoulders ---------------------

        nose = self._point(keypoints, "nose")

        head_drop = None
        head_offset = None

        if nose is not None:
            head_drop = float(
                (shoulder_center[1] - nose[1]) / shoulder_width
            )
            head_offset = float(
                (nose[0] - shoulder_center[0]) / shoulder_width
            )

        # ---- Hands -------------------------------------------------------

        wrists = {
            "left": self._point(keypoints, "left_wrist"),
            "right": self._point(keypoints, "right_wrist"),
        }

        shoulders = {
            "left": left_shoulder,
            "right": right_shoulder,
        }

        reliable_wrists = {
            side: point for side, point in wrists.items()
            if point is not None
        }

        hand_raised = None
        hands_near_face = None

        if reliable_wrists:

            hand_raised = 0

            for side, wrist in reliable_wrists.items():

                height_above_shoulder = (
                    shoulders[side][1] - wrist[1]
                ) / shoulder_width

                if height_above_shoulder > self.HAND_RAISED_MARGIN:
                    hand_raised = 1

            if nose is not None:

                hands_near_face = 0

                for wrist in reliable_wrists.values():

                    distance = np.linalg.norm(wrist - nose) / shoulder_width

                    if distance < self.HAND_NEAR_FACE_DIST:
                        hands_near_face = 1

        return {
            "shoulder_tilt": shoulder_tilt,
            "head_drop": head_drop,
            "head_offset": head_offset,
            "hand_raised": hand_raised,
            "hands_near_face": hands_near_face,
        }
