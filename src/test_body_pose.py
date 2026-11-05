from paths import TEST_VIDEO
import cv2

from body_pose import BodyPoseEstimator

file_path = str(TEST_VIDEO)

pose_estimator = BodyPoseEstimator()

cap = cv2.VideoCapture(file_path)

if not cap.isOpened():
    print("Error: Unable to open camera.")
    exit()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = pose_estimator.estimate(frame)
    persons = pose_estimator.extract_keypoints(results)

    for i, person in enumerate(persons):

        print(
            f"Person {i}: "
            f"{len(person['keypoints'])} keypoints"
        )

    annotated = pose_estimator.draw(
        frame,
        results
    )

    cv2.imshow(
        "Body Pose Test",
        annotated
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()