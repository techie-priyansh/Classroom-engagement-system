from paths import CALIBRATION_DIR
import cv2
import numpy as np
from pathlib import Path


# --------------------------------------------------
# Paths
# --------------------------------------------------

IMAGE_DIR = CALIBRATION_DIR / "images"

CALIBRATION_FILE = CALIBRATION_DIR / "results" / "droidcam_calibration.npz"


# --------------------------------------------------
# Chessboard configuration
# --------------------------------------------------

# Number of inner corners
PATTERN_SIZE = (9, 6)

# Physical size of one square
# Must match the value used during calibration
SQUARE_SIZE = 1.0


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    # --------------------------------------------------
    # Load calibration
    # --------------------------------------------------

    data = np.load(CALIBRATION_FILE)

    camera_matrix = data["camera_matrix"]
    distortion_coeffs = data["distortion_coeffs"]

    # --------------------------------------------------
    # Prepare 3D chessboard points
    # --------------------------------------------------

    object_points_template = np.zeros(
        (PATTERN_SIZE[0] * PATTERN_SIZE[1], 3),
        np.float32
    )

    object_points_template[:, :2] = np.mgrid[
        0:PATTERN_SIZE[0],
        0:PATTERN_SIZE[1]
    ].T.reshape(-1, 2)

    object_points_template *= SQUARE_SIZE

    image_errors = []

    # --------------------------------------------------
    # Process calibration images
    # --------------------------------------------------

    image_paths = sorted(IMAGE_DIR.glob("calib_*.jpg"))

    print(f"Found {len(image_paths)} calibration images.\n")

    for image_path in image_paths:

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"[SKIP] {image_path.name}")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect chessboard
        success, corners = cv2.findChessboardCornersSB(
            gray,
            PATTERN_SIZE
        )

        if not success:
            print(f"[FAIL] {image_path.name}")
            continue

        # --------------------------------------------------
        # Estimate camera pose for this image
        # --------------------------------------------------

        success, rvec, tvec = cv2.solvePnP(
            object_points_template,
            corners,
            camera_matrix,
            distortion_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            print(f"[POSE FAIL] {image_path.name}")
            continue

        # --------------------------------------------------
        # Project 3D points back into image
        # --------------------------------------------------

        projected_points, _ = cv2.projectPoints(
            object_points_template,
            rvec,
            tvec,
            camera_matrix,
            distortion_coeffs
        )

        projected_points = projected_points.reshape(-1, 2)
        observed_points = corners.reshape(-1, 2)

        # --------------------------------------------------
        # Calculate reprojection error
        # --------------------------------------------------

        errors = np.linalg.norm(
            observed_points - projected_points,
            axis=1
        )

        mean_error = np.mean(errors)
        rms_error = np.sqrt(np.mean(errors ** 2))
        max_error = np.max(errors)

        image_errors.append(mean_error)

        print(
            f"{image_path.name:15s} "
            f"Mean: {mean_error:.4f} px   "
            f"RMS: {rms_error:.4f} px   "
            f"Max: {max_error:.4f} px"
        )

    # --------------------------------------------------
    # Overall statistics
    # --------------------------------------------------

    if len(image_errors) == 0:
        print("\nNo valid images.")
        return

    print("\n========================================")
    print("REPROJECTION VALIDATION")
    print("========================================")

    print(
        f"Mean image error : "
        f"{np.mean(image_errors):.4f} px"
    )

    print(
        f"Maximum image error : "
        f"{np.max(image_errors):.4f} px"
    )

    print(
        f"Minimum image error : "
        f"{np.min(image_errors):.4f} px"
    )

    print(
        f"Std deviation : "
        f"{np.std(image_errors):.4f} px"
    )


if __name__ == "__main__":
    main()