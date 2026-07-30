from dataclasses import dataclass, field
import time

@dataclass
class Student:
    student_id: int
    frames_seen: int =0
    first_seen: float = field(default_factory= time.time)
    last_seen: float = field(default_factory= time.time)
    current_box: list | None= None
    engagement_score: float = 0.0

    def update(self, box):
        self.frames_seen+=1
        self.last_seen= time.time()
        self.current_box= box
    @property
    def visible_time(self):
        return round(self.last_seen - self.first_seen, 2)

    # def get_student(self, track_id):
    #     return self.students.get(track_id)

    landmarks_points = None
    head_pose = None
