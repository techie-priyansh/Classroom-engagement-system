from paths import CALIBRATION_DIR
import cv2
import numpy as np
from pathlib import Path


# --------------------------------------------------
# Configuration
# --------------------------------------------------

IMAGE_DIR = CALIBRATION_DIR / "images"
OUTPUT_DIR = CALIBRATION_DIR / "results"

CHECKERBOARD = (9, 6)       # internal corners
SQUARE_SIZE = 25.0          # mm


# --------------------------------------------------
# Main calibration
# --------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(IMAGE_DIR.glob("*.jpg"))

    if len(image_paths) == 0:
        print("No calibration images found.")
        return

    print(f"Found {len(image_paths)} calibration images.")

    # --------------------------------------------------
    # Create 3D object points
    # --------------------------------------------------

    object_points_template = np.zeros(
        (CHECKERBOARD[0] * CHECKERBOARD[1], 3),
        np.float32
    )

    object_points_template[:, :2] = np.mgrid[
        0:CHECKERBOARD[0],
        0:CHECKERBOARD[1]
    ].T.reshape(-1, 2)

    object_points_template *= SQUARE_SIZE

    object_points = []
    image_points = []

    image_size = None

    # --------------------------------------------------
    # Detect chessboard corners
    # --------------------------------------------------

    for image_path in image_paths:

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"[ERROR] Could not read: {image_path}")
            continue

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        if image_size is None:
            image_size = gray.shape[::-1]

        found, corners = cv2.findChessboardCornersSB(
            gray,
            CHECKERBOARD
        )

        if found:

            object_points.append(
                object_points_template.copy()
            )

            image_points.append(corners)

            print(f"[OK]   {image_path.name}")

        else:

            print(f"[FAIL] {image_path.name}")

    # --------------------------------------------------
    # Check number of valid images
    # --------------------------------------------------

    print()
    print(
        f"Valid calibration images: "
        f"{len(object_points)}/{len(image_paths)}"
    )

    if len(object_points) < 10:
        print("Not enough valid images for calibration.")
        return

    # --------------------------------------------------
    # Camera calibration
    # --------------------------------------------------

    print("\nRunning camera calibration...")

    rms_error, camera_matrix, dist_coeffs, rvecs, tvecs = (
        cv2.calibrateCamera(
            object_points,
            image_points,
            image_size,
            None,
            None
        )
    )

    # --------------------------------------------------
    # Print results
    # --------------------------------------------------

    print("\n========== CALIBRATION RESULTS ==========")

    print("\nCamera Matrix (K):")
    print(camera_matrix)

    print("\nDistortion Coefficients (D):")
    print(dist_coeffs)

    print(f"\nRMS Reprojection Error: {rms_error:.4f} pixels")

    # --------------------------------------------------
    # Calculate mean reprojection error
    # --------------------------------------------------

    total_error = 0

    for i in range(len(object_points)):

        projected_points, _ = cv2.projectPoints(
            object_points[i],
            rvecs[i],
            tvecs[i],
            camera_matrix,
            dist_coeffs
        )

        error = cv2.norm(
            image_points[i],
            projected_points,
            cv2.NORM_L2
        ) / len(projected_points)

        total_error += error

    mean_error = total_error / len(object_points)

    print(
        f"Mean Reprojection Error: "
        f"{mean_error:.4f} pixels"
    )

    # --------------------------------------------------
    # Save calibration parameters
    # --------------------------------------------------

    output_file = OUTPUT_DIR / "droidcam_calibration.npz"

    np.savez(
        output_file,
        camera_matrix=camera_matrix,
        distortion_coeffs=dist_coeffs,
        image_width=image_size[0],
        image_height=image_size[1],
        rms_error=rms_error,
        mean_reprojection_error=mean_error
    )

    print(f"\nCalibration saved to:")
    print(output_file)


if __name__ == "__main__":
    main()