import os

# Must be set BEFORE the first VideoCapture is created.
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import sys
import time
import threading
from urllib.parse import quote

import cv2
from onvif import ONVIFCamera


# ============================================================
# CAMERA CONFIGURATION
# ============================================================

IP = "192.168.1.123"
ONVIF_PORT = 2000

USER = "admin"
PASSWORD = "admin"

RTSP_NEEDS_AUTH = False

if RTSP_NEEDS_AUTH:
    RTSP_URL = f"rtsp://{quote(USER)}:{quote(PASSWORD)}@{IP}:554/live/av0"
else:
    RTSP_URL = f"rtsp://{IP}:554/live/av0"


# ============================================================
# SWEEP CONFIGURATION  (edit these)
# ============================================================
# All values are in the SOFTWARE coordinate system, measured from the
# RESET pose (0, 0, 0). Pan: + = right, - = left. Tilt: - = down.
#
# Defaults are the three poses from your screenshots:
#   right  -> PAN +0.080     middle -> PAN -0.120     left -> PAN -0.320
#   tilt   -> -0.120         zoom   -> 0.120
#
# Values should be multiples of STEP (0.04). Others are rounded to the
# nearest step and a warning is printed.

R_MAX = +0.110        # Region 1: pan limit on the right
MIDDLE_MAX = -0.020   # Region 2: pan in the middle
L_MAX = -0.140        # Region 3: pan limit on the left

T_MAX = -0.060        # tilt, reached together with Region 1, then held
Z_MAX = 0.100         # zoom, reached together with Region 1, then held

WAIT_SECS = 5.0      # hold time at each region

SHOW_VIDEO = True    # False = run with no window (camera moves only)


# ============================================================
# PTZ MOTION CONSTANTS (same as ptz_software_position.py)
# ============================================================

MOVE_SPEED = 0.8          # ONVIF velocity used for every pulse
PULSE_SECS = 0.05         # length of every pulse
POS_UNITS_PER_SEC = 0.25  # calibration: software units per second at velocity 1.0

# Software distance covered by ONE pulse. Every move in this script is a
# whole number of identical pulses, i.e. exactly what one key press did in
# your keyboard script. So the same numbers give the same physical motion.
STEP = MOVE_SPEED * PULSE_SECS * POS_UNITS_PER_SEC   # = 0.01


def sign(n):
    return (n > 0) - (n < 0)


def to_steps(value, name):
    steps = round(value / STEP)
    if abs(value - steps * STEP) > 1e-6:
        print(f"WARNING: {name}={value} is not a multiple of {STEP:.2f}; "
              f"using {steps * STEP:+.2f}")
    return steps


# ============================================================
# LATEST-FRAME RTSP READER
# ============================================================

class LatestFrame:

    def __init__(self, url):
        self.url = url
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _open(self):
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _run(self):
        cap = self._open()
        fails = 0

        while self.running:
            if not cap.isOpened():
                print("RTSP not open, retrying...")
                cap.release()
                time.sleep(1)
                cap = self._open()
                continue

            ok, frame = cap.read()

            if ok:
                fails = 0
                with self.lock:
                    self.frame = frame
            else:
                fails += 1
                if fails > 30:
                    print("RTSP stream lost, reconnecting...")
                    cap.release()
                    time.sleep(1)
                    cap = self._open()
                    fails = 0

        cap.release()

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False
        self.thread.join(timeout=2)


# ============================================================
# PTZ CONTROLLER (ContinuousMove pulses + software position)
# ============================================================

class PTZController:

    def __init__(self, ptz, token):
        self.ptz = ptz
        self.token = token

        # Software position kept as INTEGER step counts (no float drift).
        self.pan_steps = 0
        self.tilt_steps = 0
        self.zoom_steps = 0
        self.lock = threading.Lock()

        self.abort = threading.Event()
        self.status = "Starting..."

    # ---------------- state ----------------

    def get_state(self):
        with self.lock:
            return (self.pan_steps * STEP,
                    self.tilt_steps * STEP,
                    self.zoom_steps * STEP)

    def print_position(self):
        pan, tilt, zoom = self.get_state()
        print(f"PTZ Position (software) -> "
              f"Pan: {pan:+.3f} | Tilt: {tilt:+.3f} | Zoom: {zoom:.3f}")

    # ---------------- low level ----------------

    def stop(self):
        try:
            self.ptz.Stop({
                "ProfileToken": self.token,
                "PanTilt": True,
                "Zoom": True,
            })
        except Exception as e:
            print(f"PTZ stop error: {e}")

    def pulse(self, dp, dt, dz):
        """One pulse. dp/dt/dz are -1, 0 or +1 (direction per axis).
        Raises on a failed move so the sweep can stop safely."""
        request = self.ptz.create_type("ContinuousMove")
        request.ProfileToken = self.token
        request.Velocity = {
            "PanTilt": {"x": dp * MOVE_SPEED, "y": dt * MOVE_SPEED},
            "Zoom": {"x": dz * MOVE_SPEED},
        }

        self.ptz.ContinuousMove(request)
        time.sleep(PULSE_SECS)
        self.stop()

        with self.lock:
            self.pan_steps += dp
            self.tilt_steps += dt
            self.zoom_steps += dz

    # ---------------- high level ----------------

    def move_to(self, pan=None, tilt=None, zoom=None):
        """Move to a software position. An axis left as None is held fixed.
        All axes that need to move do so at the same time."""
        with self.lock:
            n_pan = 0 if pan is None else to_steps(pan, "pan") - self.pan_steps
            n_tilt = 0 if tilt is None else to_steps(tilt, "tilt") - self.tilt_steps
            n_zoom = 0 if zoom is None else to_steps(zoom, "zoom") - self.zoom_steps

        while (n_pan or n_tilt or n_zoom) and not self.abort.is_set():
            dp, dt, dz = sign(n_pan), sign(n_tilt), sign(n_zoom)
            self.pulse(dp, dt, dz)
            n_pan -= dp
            n_tilt -= dt
            n_zoom -= dz

        self.print_position()

    def hold(self, secs, label):
        """Wait `secs` seconds (with a countdown on screen). Returns False if aborted."""
        end = time.monotonic() + secs
        while True:
            left = end - time.monotonic()
            if left <= 0:
                return True
            self.status = f"{label} - holding {left:.1f}s"
            if self.abort.wait(min(0.1, left)):
                return False


# ============================================================
# THE SWEEP
# ============================================================

def run_sweep(c):
    """Reset -> Region 1 (right + tilt + zoom) -> Region 2 (middle)
    -> Region 3 (left). Tilt and zoom stay fixed after Region 1."""
    try:
        c.status = "Moving to Region 1 (right)"
        print("\n--- Reset -> Region 1 (right, tilt, zoom) ---")
        c.move_to(pan=R_MAX, tilt=T_MAX, zoom=Z_MAX)
        if not c.hold(WAIT_SECS, "Region 1"):
            return

        c.status = "Moving to Region 2 (middle)"
        print("\n--- Region 1 -> Region 2 (middle) ---")
        c.move_to(pan=MIDDLE_MAX)
        if not c.hold(WAIT_SECS, "Region 2"):
            return

        c.status = "Moving to Region 3 (left)"
        print("\n--- Region 2 -> Region 3 (left) ---")
        c.move_to(pan=L_MAX)
        if not c.hold(WAIT_SECS, "Region 3"):
            return

        c.status = "Sweep complete"
        print("\nSweep complete.")

    except Exception as e:
        print(f"\nSweep stopped by error: {e}")
        c.abort.set()
        c.status = "Error - sweep stopped"

    finally:
        c.stop()   # never leave the camera moving


# ============================================================
# VIDEO OVERLAY
# ============================================================

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def put_text(frame, text, x, y, scale=0.7):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 255, 0), 2, cv2.LINE_AA)


def draw_bar(frame, x, y, width, height, value, vmin, vmax, label):
    cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 255), 1)
    pos = int(clamp((value - vmin) / (vmax - vmin), 0.0, 1.0) * width)
    cv2.line(frame, (x + pos, y), (x + pos, y + height), (0, 255, 0), 3)
    cv2.putText(frame, label, (x + width + 10, y + height - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def draw_overlay(frame, c):
    pan, tilt, zoom = c.get_state()

    put_text(frame, "PTZ SWEEP (software position)", 20, 30, 0.6)
    put_text(frame, c.status, 20, 62, 0.7)
    put_text(frame, f"PAN  : {pan:+.3f}", 20, 94)
    put_text(frame, f"TILT : {tilt:+.3f}", 20, 126)
    put_text(frame, f"ZOOM : {zoom:.3f}", 20, 158)

    draw_bar(frame, 20, 177, 240, 15, pan, -1.0, 1.0, "PAN")
    draw_bar(frame, 20, 207, 240, 15, tilt, -1.0, 1.0, "TILT")
    draw_bar(frame, 20, 237, 240, 15, zoom, 0.0, 1.0, "ZOOM")


# ============================================================
# MAIN
# ============================================================

def validate_config():
    for name, v in (("R_MAX", R_MAX), ("MIDDLE_MAX", MIDDLE_MAX),
                    ("L_MAX", L_MAX), ("T_MAX", T_MAX)):
        if not -1.0 <= v <= 1.0:
            sys.exit(f"ERROR: {name}={v} must be between -1.0 and 1.0")
    if not 0.0 <= Z_MAX <= 1.0:
        sys.exit(f"ERROR: Z_MAX={Z_MAX} must be between 0.0 and 1.0")


def main():
    validate_config()

    print()
    print("Connecting to PTZ camera...")

    try:
        cam = ONVIFCamera(IP, ONVIF_PORT, USER, PASSWORD)
        media = cam.create_media_service()
        ptz = cam.create_ptz_service()

        profiles = media.GetProfiles()
        if not profiles:
            raise RuntimeError("Camera returned no ONVIF profiles.")

        token = profiles[0].token

    except Exception as e:
        print(f"ERROR: ONVIF connection failed: {e}")
        sys.exit(1)

    print("ONVIF connected.")

    controller = PTZController(ptz, token)

    print()
    print("======================================")
    print("            PTZ AUTO SWEEP")
    print("======================================")
    print("Make sure the camera is at its RESET pose first.")
    print(f"Region 1 (right) : pan {R_MAX:+.2f}, tilt {T_MAX:+.2f}, zoom {Z_MAX:.2f}")
    print(f"Region 2 (middle): pan {MIDDLE_MAX:+.2f}")
    print(f"Region 3 (left)  : pan {L_MAX:+.2f}")
    print(f"Hold {WAIT_SECS:.0f}s at each region.   Q = abort")
    print("======================================")

    reader = None

    try:
        if SHOW_VIDEO:
            reader = LatestFrame(RTSP_URL).start()
            cv2.namedWindow("PTZ Camera", cv2.WINDOW_NORMAL)

            sweep = threading.Thread(target=run_sweep, args=(controller,), daemon=True)
            sweep.start()

            while sweep.is_alive():
                frame = reader.read()
                if frame is not None:
                    draw_overlay(frame, controller)
                    cv2.imshow("PTZ Camera", frame)

                key = cv2.waitKeyEx(30)

                # Q aborts. Compare the full code so arrow keys (whose low
                # byte looks like 'Q') cannot abort by accident.
                if (key & 0xFFFF) in (ord("q"), ord("Q")):
                    print("Q pressed - aborting.")
                    controller.abort.set()
                    break

                # Window X button also aborts.
                if cv2.getWindowProperty("PTZ Camera", cv2.WND_PROP_VISIBLE) < 1:
                    controller.abort.set()
                    break

            sweep.join(timeout=5)
        else:
            run_sweep(controller)

    except KeyboardInterrupt:
        print("Ctrl+C - aborting.")
        controller.abort.set()

    finally:
        controller.stop()
        if reader is not None:
            reader.stop()
        cv2.destroyAllWindows()
        print("Script finished. Camera left at:")
        controller.print_position()


if __name__ == "__main__":
    main()