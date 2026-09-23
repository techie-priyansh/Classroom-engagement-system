import cv2
import numpy as np
from pathlib import Path


# PHONE_IMAGE = Path("/home/mylinux/Desktop/MTech_Project/MTP/calibration/virtualClassroom/calib_12.jpg")
# LAPTOP_IMAGE = Path("/home/mylinux/Desktop/MTech_Project/MTP/calibration/virtualClassroom/calib_01.jpg")


# --------------------------------------------------
# Configuration
# --------------------------------------------------

# PHONE_IMAGE = Path("/home/mylinux/Desktop/MTech_Project/MTP/Classroom-engagement-system/src/calib_12.jpg")
# LAPTOP_IMAGE = Path("/home/mylinux/Desktop/MTech_Project/MTP/Classroom-engagement-system/src/calib_01.jpg")

#Paths
PHONE_IMAGE = Path("/home/mylinux/Desktop/MTech_Project/MTP/Classroom-engagement-system/src/Mobile1_train.jpg")
LAPTOP_IMAGE = Path("/home/mylinux/Desktop/MTech_Project/MTP/Classroom-engagement-system/src/Laptop1_train.jpg")

OUTPUT_DIR = Path(
    "/home/mylinux/Desktop/MTech_Project/MTP/Classroom-engagement-system/calibration/results"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Physical coordinates of the X centers
# P1 = bottom-left
# P2 = bottom-right
# P3 = top-right
# P4 = top-left
#
# Units: meters

WORLD_POINTS = np.array([
    [0.0, 0.0],
    [1.10, 0.0],
    [1.10, 1.10],
    [0.0, 1.10]
], dtype=np.float32)


# Output top-view size
OUTPUT_SIZE = 600


def select_points(image, window_name):
    """
    Manually select four X centers.

    Order:
        1. bottom-left
        2. bottom-right
        3. top-right
        4. top-left
    """

    points = []

    display = image.copy()

    def mouse_callback(event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:

            points.append([x, y])

            cv2.circle(
                display,
                (x, y),
                6,
                (0, 255, 0),
                -1
            )

            cv2.putText(
                display,
                str(len(points)),
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.imshow(window_name, display)

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print(f"\nSelect 4 points in {window_name}")
    print("Order:")
    print("1 -> bottom-left")
    print("2 -> bottom-right")
    print("3 -> top-right")
    print("4 -> top-left")
    print("Press ESC after selecting all 4 points.")

    while True:

        cv2.imshow(window_name, display)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break

        if len(points) == 4:
            print("4 points selected.")
            break

    cv2.destroyWindow(window_name)

    if len(points) != 4:
        raise RuntimeError(
            f"Expected 4 points, but selected {len(points)}."
        )

    return np.array(points, dtype=np.float32)


def create_world_points():
    """
    Convert physical coordinates in meters
    into output image coordinates.
    """

    scale = OUTPUT_SIZE / 1.10

    points = WORLD_POINTS.copy()

    # X -> image x
    points[:, 0] *= scale

    # Y -> image y
    # Flip because image y increases downward.
    points[:, 1] = OUTPUT_SIZE - points[:, 1] * scale

    return points


def process_camera(image, name):

    image_points = select_points(
        image,
        f"{name} - Select X Centers"
    )

    destination_points = create_world_points()

    H, mask = cv2.findHomography(
        image_points,
        destination_points,
        method=0
    )

    if H is None:
        raise RuntimeError(
            f"Could not calculate homography for {name}"
        )

    warped = cv2.warpPerspective(
        image,
        H,
        (OUTPUT_SIZE, OUTPUT_SIZE)
    )

    print(f"\n========== {name} ==========")
    print("Image points:")
    print(image_points)

    print("\nHomography matrix:")
    print(H)

    output_path = OUTPUT_DIR / f"{name.lower()}_topview.jpg"

    cv2.imwrite(
        str(output_path),
        warped
    )

    print(f"\nSaved top-view:")
    print(output_path)

    return H, warped


def main():

    print("Loading images...")

    phone_image = cv2.imread(str(PHONE_IMAGE))
    laptop_image = cv2.imread(str(LAPTOP_IMAGE))

    if phone_image is None:
        raise FileNotFoundError(
            f"Could not load phone image:\n{PHONE_IMAGE}"
        )

    if laptop_image is None:
        raise FileNotFoundError(
            f"Could not load laptop image:\n{LAPTOP_IMAGE}"
        )

    print("Images loaded successfully.")

    # --------------------------------------------------
    # Phone camera
    # --------------------------------------------------

    H_phone, phone_topview = process_camera(
        phone_image,
        "Phone"
    )

    # --------------------------------------------------
    # Laptop camera
    # --------------------------------------------------

    H_laptop, laptop_topview = process_camera(
        laptop_image,
        "Laptop"
    )

    # --------------------------------------------------
    # Display results
    # --------------------------------------------------

    cv2.imshow(
        "Phone Top View",
        phone_topview
    )

    cv2.imshow(
        "Laptop Top View",
        laptop_topview
    )

    print("\nPress any key to exit.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()