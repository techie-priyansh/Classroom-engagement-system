# Classroom-engagement-system
camera.py
    ↓
detector.py          → YOLO person detection
    ↓
tracker.py           → ByteTrack
    ↓
student_manager.py   → student IDs / state
    ↓
face_mesh.py         → MediaPipe 478 landmarks
    ↓
head_pose.py         → yaw / pitch / roll
    ↓
eye_analyser.py      → EAR / eye state
    ↓
gaze_estimator.py    → LEFT / RIGHT / UP / DOWN / FORWARD
    ↓
body_pose.py         → 17 body keypoints
    ↓
posture_analyser.py  → posture features
    ↓
pipeline.py