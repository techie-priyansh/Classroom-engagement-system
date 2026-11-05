"""Unit tests for Student / StudentManager (no camera needed).

Run from the src folder:  python test_student_manager.py
"""

from student import Student
from student_manager import StudentManager


def manager():
    return StudentManager(verbose=False)


def test_fields_are_per_student():
    a, b = Student(1), Student(2)
    a.left_ear = 0.3
    assert b.left_ear is None


def test_video_time_is_used():
    m = manager()
    m.update_student(5, [0, 0, 10, 10], frame_number=100, timestamp=4.0)
    s = m.update_student(5, [0, 0, 10, 10], frame_number=125, timestamp=5.0)
    assert s.frames_seen == 2
    assert s.first_frame == 100 and s.last_frame == 125
    assert s.visible_time == 1.0


def test_measurements_do_not_go_stale():
    m = manager()
    s = m.update_student(1, [0, 0, 1, 1], 1, 0.0)
    s.head_pose = (10.0, 5.0, 0.0)
    s.avg_ear = 0.28
    s.gaze_direction = "FORWARD"

    s = m.update_student(1, [0, 0, 1, 1], 2, 0.04)   # face lost on this frame
    assert s.head_pose is None and s.avg_ear is None and s.gaze_direction is None


def test_active_students_are_those_seen_now():
    m = manager()
    m.update_student(1, [0, 0, 1, 1], 1, 0.0)
    m.update_student(2, [0, 0, 1, 1], 1, 0.0)
    m.update_student(1, [0, 0, 1, 1], 2, 0.04)
    assert [s.student_id for s in m.active_students(2)] == [1]


def test_wall_clock_fallback_and_reset():
    m = manager()
    m.update_student(1, [0, 0, 1, 1])
    assert m.get_student(1).frames_seen == 1
    m.reset()
    assert m.get_students() == {} and m.get_student(1) is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
