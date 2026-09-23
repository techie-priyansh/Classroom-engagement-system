import cv2
from pathlib import Path


#CAMERA_DEVICE = "/dev/video2"
OUTPUT_DIR = Path("calibration/virtualClassroom")
CAMERA_DEVICE = 0


def get_next_count(output_dir):

    existing_files = output_dir.glob("calib_*.jpg")

    numbers = []

    for file in existing_files:
        try:
            number = int(file.stem.split("_")[-1])
            numbers.append(number)
        except ValueError:
            pass

    return max(numbers, default=0) + 1


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(CAMERA_DEVICE)

    if not cap.isOpened():
        print(f"Error: Could not open {CAMERA_DEVICE}")
        return

    # Automatically determine the next image number
    count = get_next_count(OUTPUT_DIR)

    print("Calibration image capture")
    print("-------------------------")
    print(f"Camera : {CAMERA_DEVICE}")
    print(f"Output : {OUTPUT_DIR}")
    print(f"Starting from: calib_{count:02d}.jpg")
    print()
    print("SPACE → capture image")
    print("Q     → quit")

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to read frame.")
            break

        cv2.imshow(
            "DroidCam Calibration Capture",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):

            filename = OUTPUT_DIR / f"calib_{count:02d}.jpg"

            success = cv2.imwrite(
                str(filename),
                frame
            )

            if success:
                print(f"Saved: {filename}")
                count += 1
            else:
                print(f"Error: Could not save {filename}")

        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\nNext available image number: {count}")


if __name__ == "__main__":
    main()