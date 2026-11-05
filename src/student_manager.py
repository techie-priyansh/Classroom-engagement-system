from student import Student


class StudentManager:
    """Keeps one Student per tracker ID for the current camera region."""

    def __init__(self, verbose=True):
        self.students = {}
        self.verbose = verbose

    def update_student(self, track_id, box, frame_number=None, timestamp=None):
        """
        Register a detection. Pass the frame number and the video time when
        processing a video; without them the wall clock is used (live use).
        """

        student = self.students.get(track_id)

        if student is None:

            if self.verbose:
                print(f"New Student Registered -> ID {track_id}")

            student = Student(track_id)
            self.students[track_id] = student

        student.update(box, frame_number, timestamp)

        return student

    def get_student(self, track_id):
        return self.students.get(track_id)

    def get_students(self):
        return self.students

    def active_students(self, frame_number):
        """Students detected on this frame (not just earlier ones)."""

        return [s for s in self.students.values() if s.last_frame == frame_number]

    def print_summary(self):
        print("\n" + "=" * 50)
        print("Current Students")
        print("=" * 50)

        for student in self.students.values():
            print(
                f"ID: {student.student_id} | "
                f"Frames: {student.frames_seen} | "
                f"Visible: {student.visible_time:.2f}s"
            )

    def reset(self):
        """Clear student records when entering a new PTZ region."""

        self.students.clear()

        if self.verbose:
            print("StudentManager reset successfully.")
