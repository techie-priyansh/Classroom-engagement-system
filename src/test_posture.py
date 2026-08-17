import cv2

from body_pose import BodyPoseEstimator
from posture_analyser import PostureAnalyser


def main():

    pose_estimator = BodyPoseEstimator()
    posture_analyser = PostureAnalyser()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to open camera.")
        return

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # Body pose
        results = pose_estimator.estimate(frame)

        # Extract keypoints
        persons = pose_estimator.extract_keypoints(results)

        # Process each person
        for i, person in enumerate(persons):

            # Convert to named keypoints
            keypoints = pose_estimator.get_named_keypoints(person)

            # Posture features
            features = posture_analyser.analyse(keypoints)

            if features is None:
                continue

            torso_angle = features["torso_angle"]
            left_knee = features["left_knee_angle"]
            right_knee = features["right_knee_angle"]

            print(
                f"Person {i} | "
                f"Torso: {torso_angle:.1f}° | "
                f"Left Knee: {left_knee:.1f}° | "
                f"Right Knee: {right_knee:.1f}°"
            )

        # Draw pose
        annotated = pose_estimator.draw(
            frame,
            results
        )

        cv2.imshow(
            "Posture Test",
            annotated
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()