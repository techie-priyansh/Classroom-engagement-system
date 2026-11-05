"""One tracked student: how long they have been seen, and the measurements
of the CURRENT frame.

Two things were wrong in the old class:
  * the measurement fields (`left_ear`, `eye_state`, ...) had no type
    annotation, so they were shared class attributes, not per-student
    fields; `landmarks_points` was misspelt (the pipeline writes
    `landmark_points`); `avg_ear` and `gaze_direction` did not exist;
  * measurements stayed on the object after the frame they belonged to,
    so a student whose face was lost still showed the head pose of an
    earlier frame (stale state).

Now every field is declared, and `update()` clears the per-frame
measurements, so a field that is not None was measured on THIS frame.
Times are video times when the caller gives them (a video is processed
faster than real time, so the wall clock would be meaningless).
"""

import time
from dataclasses import dataclass, field


@dataclass
class Student:
    student_id: int

    # ---- presence ----
    frames_seen: int = 0
    first_frame: int | None = None
    last_frame: int | None = None
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    current_box: list | None = None

    # ---- measurements of the current frame (None = not measured now) ----
    landmark_points: object = None
    head_pose: tuple | None = None       # (yaw, pitch, roll)
    left_ear: float | None = None
    right_ear: float | None = None
    avg_ear: float | None = None
    gaze_direction: str | None = None

    engagement_score: float | None = None  # set by engagement_score.py, not per frame

    def update(self, box, frame_number=None, timestamp=None):
        """Register a new detection and clear the previous frame's measurements."""

        if self.frames_seen == 0 and timestamp is not None:
            self.first_seen = timestamp

        if self.first_frame is None:
            self.first_frame = frame_number

        self.frames_seen += 1
        self.last_frame = frame_number
        self.last_seen = time.time() if timestamp is None else timestamp
        self.current_box = box

        self.landmark_points = None
        self.head_pose = None
        self.left_ear = None
        self.right_ear = None
        self.avg_ear = None
        self.gaze_direction = None

    @property
    def visible_time(self):
        return round(self.last_seen - self.first_seen, 2)
