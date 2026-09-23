import cv2
import numpy as np


H_PHONE = np.array([
    [1.29762623e+00, 3.48293976e-01, -4.53333891e+02],
    [-2.12696304e-02, 2.16099444e+00, -2.69907355e+02],
    [-3.16365594e-05, 9.59370006e-04, 1.00000000e+00]
])

H_LAPTOP = np.array([
    [1.87770815e+00, 3.19471178e-01, -2.51123905e+02],
    [2.14569647e-01, 3.57616079e+00, -2.72932592e+02],
    [-2.12891062e-05, 1.96243164e-03, 1.00000000e+00]
])


PHONE_IMAGE = "src/Mobile2_test.jpg"
LAPTOP_IMAGE = "src/Laptop2_test.jpg"

OUTPUT_SIZE = 600
WORLD_SIZE = 1.10

TRUE_POSITION = np.array([0.55, 0.55])


def select_point(image, window_name):

    point = []
    display = image.copy()

    def mouse_callback(event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:

            point.append([x, y])

            cv2.circle(
                display,
                (x, y),
                6,
                (0, 255, 0),
                -1
            )

            cv2.imshow(window_name, display)

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print(f"\nSelect the CENTER of the box in {window_name}")

    while True:

        cv2.imshow(window_name, display)

        key = cv2.waitKey(1) & 0xFF

        if len(point) == 1:
            break

        if key == 27:
            break

    cv2.destroyWindow(window_name)

    if len(point) != 1:
        raise RuntimeError("Point selection failed.")

    return np.array(point[0], dtype=np.float32)


def image_to_world(point, H):

    # Convert image point to homogeneous coordinates
    image_point = np.array([
        point[0],
        point[1],
        1.0
    ])

    # Camera image -> 600x600 top-view
    topview_point = H @ image_point

    # Normalize homogeneous coordinate
    topview_point /= topview_point[2]

    x_top = topview_point[0]
    y_top = topview_point[1]

    # 600x600 pixels -> meters
    X = x_top * WORLD_SIZE / OUTPUT_SIZE

    # Undo the Y-axis flip used in create_world_points()
    Y = (OUTPUT_SIZE - y_top) * WORLD_SIZE / OUTPUT_SIZE

    return np.array([X, Y])


def main():

    phone = cv2.imread(PHONE_IMAGE)
    laptop = cv2.imread(LAPTOP_IMAGE)

    if phone is None:
        raise FileNotFoundError(PHONE_IMAGE)

    if laptop is None:
        raise FileNotFoundError(LAPTOP_IMAGE)

    print("Images loaded successfully.")

    # ------------------------------------------
    # Select box center in both cameras
    # ------------------------------------------

    phone_point = select_point(
        phone,
        "Phone - Box Center"
    )

    laptop_point = select_point(
        laptop,
        "Laptop - Box Center"
    )

    # ------------------------------------------
    # Convert to world coordinates
    # ------------------------------------------

    phone_world = image_to_world(
        phone_point,
        H_PHONE
    )

    laptop_world = image_to_world(
        laptop_point,
        H_LAPTOP
    )

    # ------------------------------------------
    # Calculate errors
    # ------------------------------------------

    phone_error = np.linalg.norm(
        phone_world - TRUE_POSITION
    )

    laptop_error = np.linalg.norm(
        laptop_world - TRUE_POSITION
    )

    cross_camera_error = np.linalg.norm(
        phone_world - laptop_world
    )

    # ------------------------------------------
    # Results
    # ------------------------------------------

    print("\n===================================")
    print("CROSS-CAMERA LOCALIZATION")
    print("===================================")

    print("\nImage points:")
    print(f"Phone : {phone_point}")
    print(f"Laptop: {laptop_point}")

    print("\nEstimated world positions:")

    print(
        f"Phone : "
        f"X={phone_world[0]:.4f} m, "
        f"Y={phone_world[1]:.4f} m"
    )

    print(
        f"Laptop: "
        f"X={laptop_world[0]:.4f} m, "
        f"Y={laptop_world[1]:.4f} m"
    )

    print("\nKnown box position:")
    print("X=0.5500 m, Y=0.5500 m")

    print("\nLocalization errors:")

    print(
        f"Phone absolute error : "
        f"{phone_error:.4f} m"
    )

    print(
        f"Laptop absolute error: "
        f"{laptop_error:.4f} m"
    )

    print(
        f"Cross-camera error   : "
        f"{cross_camera_error:.4f} m"
    )


if __name__ == "__main__":
    main()