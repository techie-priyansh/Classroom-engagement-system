import cv2
from pathlib import Path


#CAMERA_DEVICE = "/dev/video2"
OUTPUT_DIR = Path("calibration/virtualClassroom")
CAMERA_DEVICE = 2


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(CAMERA_DEVICE)

    if not cap.isOpened():
        print(f"Error: Could not open {CAMERA_DEVICE}")
        return

    count = 10

    print("Calibration image capture")
    print("-------------------------")
    print("SPACE → capture image")
    print("Q     → quit")

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to read frame.")
            break

        cv2.imshow("DroidCam Calibration Capture", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):

            filename = OUTPUT_DIR / f"calib_{count:02d}.jpg"

            cv2.imwrite(
                str(filename),
                frame
            )

            print(f"Saved: {filename}")

            count += 1

        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\nCaptured {count} images.")


if __name__ == "__main__":
    main()