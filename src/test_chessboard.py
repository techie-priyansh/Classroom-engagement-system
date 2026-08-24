import cv2

IMAGE_PATH = "/home/mylinux/Desktop/MTech_Project/MTP/calibration/images/calib_04.jpg"

CHECKERBOARD = (9, 6)


def main():

    image = cv2.imread(IMAGE_PATH)

    if image is None:
        print("Error: Could not read image.")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # found, corners = cv2.findChessboardCorners(
    #     gray,
    #     CHECKERBOARD,
    #     cv2.CALIB_CB_ADAPTIVE_THRESH
    #     + cv2.CALIB_CB_NORMALIZE_IMAGE
    # )
    found, corners = cv2.findChessboardCornersSB(
        gray,
        CHECKERBOARD
    )
    print(f"Chessboard detected: {found}")

    if found:

        cv2.drawChessboardCorners(
            image,
            CHECKERBOARD,
            corners,
            found
        )

    cv2.imshow("Chessboard Detection", image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()