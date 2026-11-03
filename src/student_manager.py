from student import Student

class StudentManager:
    def __init__(self):
        self.students= {}

    def update_student(self, track_id, box):
        if track_id not in self.students:
            print(f"New Student Registered -> ID {track_id}")
            self.students[track_id] = Student(track_id)

        self.students[track_id].update(box)


    def get_students(self):
        return self.students

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
        print("StudentManager reset successfully.")