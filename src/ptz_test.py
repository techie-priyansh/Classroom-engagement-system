import cv2
import subprocess
import time
import threading

# ============================================================
# CONFIGURATION
# ============================================================

CAMERA = "/dev/video2"

WIDTH = 1920
HEIGHT = 1080
FPS = 30

OUTPUT_FILE = "ptz_recording.avi"

# PTZ speed
PAN_SPEED = 50
TILT_SPEED = 50

# Zoom step
ZOOM_STEP = 1
ZOOM_MIN = 0
ZOOM_MAX = 2

# Current zoom
zoom = 1

# ============================================================
# PTZ CONTROL
# ============================================================

def set_ptz_control(control, value):
    """
    Set a V4L2 PTZ control using v4l2-ctl.
    """

    command = [
        "v4l2-ctl",
        "-d", CAMERA,
        f"--set-ctrl={control}={value}"
    ]

    subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )


def stop_ptz():
    """
    Stop both pan and tilt movement.
    """

    set_ptz_control("pan_speed", 0)
    set_ptz_control("tilt_speed", 0)


# ============================================================
# PTZ MOVEMENT
# ============================================================

def move_ptz(pan=0, tilt=0):

    set_ptz_control("pan_speed", pan)
    set_ptz_control("tilt_speed", tilt)


# ============================================================
# CENTER CAMERA
# ============================================================

def center_camera():

    set_ptz_control("pan_absolute", 0)
    set_ptz_control("tilt_absolute", 0)

    print("Camera centered")


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(CAMERA, cv2.CAP_V4L2)

if not cap.isOpened():
    print("ERROR: Cannot open camera:", CAMERA)
    exit()


cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_FPS, FPS)


# ============================================================
# CHECK ACTUAL CAMERA FORMAT
# ============================================================

actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
actual_fps = cap.get(cv2.CAP_PROP_FPS)

print("--------------------------------------------")
print("Camera opened")
print(f"Resolution : {actual_width} x {actual_height}")
print(f"FPS        : {actual_fps}")
print("--------------------------------------------")


# ============================================================
# VIDEO WRITER
# ============================================================

fourcc = cv2.VideoWriter_fourcc(*"XVID")

writer = cv2.VideoWriter(
    OUTPUT_FILE,
    fourcc,
    FPS,
    (actual_width, actual_height)
)

if not writer.isOpened():
    print("ERROR: Cannot create video file")
    cap.release()
    exit()


# ============================================================
# INITIALIZATION
# ============================================================

stop_ptz()

time.sleep(0.5)

print()
print("============================================")
print("       PTZ CAMERA + VIDEO RECORDING")
print("============================================")
print()
print("Keyboard controls:")
print()
print("        W       Tilt Up")
print("        S       Tilt Down")
print()
print("        A       Pan Left")
print("        D       Pan Right")
print()
print("        +       Zoom In")
print("        -       Zoom Out")
print()
print("        C       Center Camera")
print("        SPACE   Stop PTZ")
print("        Q       Quit + Save Recording")
print()
print("============================================")
print()


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Frame capture failed")
        break

    # --------------------------------------------------------
    # DISPLAY STATUS
    # --------------------------------------------------------

    cv2.putText(
        frame,
        "PTZ Recording",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Zoom: {zoom}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "W/S: Tilt   A/D: Pan   +/-: Zoom   C: Center   SPACE: Stop   Q: Quit",
        (20, actual_height - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    # --------------------------------------------------------
    # RECORD
    # --------------------------------------------------------

    writer.write(frame)

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    cv2.imshow("AdonisAV PTZ Camera", frame)

    # --------------------------------------------------------
    # KEYBOARD
    # --------------------------------------------------------

    key = cv2.waitKey(1) & 0xFF

    # --------------------------------------------------------
    # QUIT
    # --------------------------------------------------------

    if key == ord("q"):

        stop_ptz()
        break

    # --------------------------------------------------------
    # STOP
    # --------------------------------------------------------

    elif key == 32:  # SPACE

        stop_ptz()
        print("PTZ stopped")

    # --------------------------------------------------------
    # CENTER
    # --------------------------------------------------------

    elif key == ord("c"):

        stop_ptz()
        center_camera()

    # --------------------------------------------------------
    # PAN LEFT
    # --------------------------------------------------------

    elif key == ord("a"):

        set_ptz_control(
            "pan_speed",
            -PAN_SPEED
        )

        set_ptz_control(
            "tilt_speed",
            0
        )

        print("Pan LEFT")

    # --------------------------------------------------------
    # PAN RIGHT
    # --------------------------------------------------------

    elif key == ord("d"):

        set_ptz_control(
            "pan_speed",
            PAN_SPEED
        )

        set_ptz_control(
            "tilt_speed",
            0
        )

        print("Pan RIGHT")

    # --------------------------------------------------------
    # TILT UP
    # --------------------------------------------------------

    elif key == ord("w"):

        set_ptz_control(
            "pan_speed",
            0
        )

        set_ptz_control(
            "tilt_speed",
            TILT_SPEED
        )

        print("Tilt UP")

    # --------------------------------------------------------
    # TILT DOWN
    # --------------------------------------------------------

    elif key == ord("s"):

        set_ptz_control(
            "pan_speed",
            0
        )

        set_ptz_control(
            "tilt_speed",
            -TILT_SPEED
        )

        print("Tilt DOWN")

    # --------------------------------------------------------
    # ZOOM IN
    # --------------------------------------------------------

    elif key in (ord("+"), ord("=")):

        zoom += ZOOM_STEP

        if zoom > ZOOM_MAX:
            zoom = ZOOM_MAX

        set_ptz_control(
            "zoom_absolute",
            zoom
        )

        print(f"Zoom IN -> {zoom}")

    # --------------------------------------------------------
    # ZOOM OUT
    # --------------------------------------------------------

    elif key == ord("-"):

        zoom -= ZOOM_STEP

        if zoom < ZOOM_MIN:
            zoom = ZOOM_MIN

        set_ptz_control(
            "zoom_absolute",
            zoom
        )

        print(f"Zoom OUT -> {zoom}")


# ============================================================
# CLEANUP
# ============================================================

stop_ptz()

cap.release()
writer.release()

cv2.destroyAllWindows()

print()
print("============================================")
print("Recording stopped")
print(f"Saved to: {OUTPUT_FILE}")
print("============================================")