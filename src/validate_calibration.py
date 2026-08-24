import cv2
import numpy as np
from pathlib import Path


# --------------------------------------------------
# Paths
# --------------------------------------------------

CALIBRATION_FILE = Path(
    "/home/mylinux/Desktop/MTech_Project/MTP/calibration/results/"
    "droidcam_calibration.npz"
)

IMAGE_FILE = Path(
    "/home/mylinux/Desktop/MTech_Project/MTP/Classroom-engagement-system/"
    "src/validation.jpg"
)

OUTPUT_FILE = Path(
    "/home/mylinux/Desktop/MTech_Project/MTP/calibration/results/"
    "validation_undistorted.jpg"
)


def main():

    # --------------------------------------------------
    # Load calibration
    # --------------------------------------------------

    data = np.load(CALIBRATION_FILE)

    camera_matrix = data["camera_matrix"]
    distortion_coeffs = data["distortion_coeffs"]

    print("Camera matrix:")
    print(camera_matrix)

    print("\nDistortion coefficients:")
    print(distortion_coeffs)

    # --------------------------------------------------
    # Load validation image
    # --------------------------------------------------

    image = cv2.imread(str(IMAGE_FILE))

    if image is None:
        print("\nError: Could not load validation image.")
        return

    h, w = image.shape[:2]

    print(f"\nImage size: {w} x {h}")

    # --------------------------------------------------
    # Compute optimal camera matrix
    # --------------------------------------------------

    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        distortion_coeffs,
        (w, h),
        1,
        (w, h)
    )

    # --------------------------------------------------
    # Undistort image
    # --------------------------------------------------

    undistorted = cv2.undistort(
        image,
        camera_matrix,
        distortion_coeffs,
        None,
        new_camera_matrix
    )

    # --------------------------------------------------
    # Crop valid region
    # --------------------------------------------------

    x, y, roi_w, roi_h = roi

    if roi_w > 0 and roi_h > 0:
        undistorted_cropped = undistorted[
            y:y + roi_h,
            x:x + roi_w
        ]
    else:
        undistorted_cropped = undistorted

    # --------------------------------------------------
    # Save result
    # --------------------------------------------------

    cv2.imwrite(
        str(OUTPUT_FILE),
        undistorted_cropped
    )

    print(f"\nUndistorted image saved to:")
    print(OUTPUT_FILE)

    # --------------------------------------------------
    # Display
    # --------------------------------------------------

    cv2.imshow("Original", image)
    cv2.imshow("Undistorted", undistorted_cropped)

    print("\nPress any key to close.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()