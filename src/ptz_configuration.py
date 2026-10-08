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
# PTZ CONFIGURATION
# ============================================================

MOVE_SPEED = 0.8     # ONVIF velocity, -1.0 to 1.0
PULSE_SECS = 0.20    # duration of each movement pulse

# Calibration constant for the SOFTWARE position.
# = how far the software coordinate moves per second at velocity 1.0.
# Tune it so the on-screen numbers match reality (see notes after the code).
POS_UNITS_PER_SEC = 0.25


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


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
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # keep latency low
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
# SOFTWARE PTZ STATE
# ============================================================
# The camera never reports its position (GetStatus is not used at all),
# so this object IS the position. It starts at (0, 0, 0) and is changed
# only by the movements WE command.

class PTZState:

    def __init__(self):
        self.pan = 0.0
        self.tilt = 0.0
        self.zoom = 0.0
        self.lock = threading.Lock()

    def move_by(self, dpan=0.0, dtilt=0.0, dzoom=0.0):
        with self.lock:
            self.pan = clamp(self.pan + dpan, -1.0, 1.0)
            self.tilt = clamp(self.tilt + dtilt, -1.0, 1.0)
            self.zoom = clamp(self.zoom + dzoom, 0.0, 1.0)

    def reset(self):
        with self.lock:
            self.pan = 0.0
            self.tilt = 0.0
            self.zoom = 0.0

    def get(self):
        with self.lock:
            return self.pan, self.tilt, self.zoom


# ============================================================
# PTZ CONTROLLER (ContinuousMove only, no GetStatus)
# ============================================================

class PTZController:

    def __init__(self, ptz, token):
        self.ptz = ptz
        self.token = token
        self.state = PTZState()

        # Only one PTZ command at a time.
        self.command_lock = threading.Lock()

    # ---------------- movement ----------------

    def pulse(self, pan=0.0, tilt=0.0, zoom=0.0, secs=PULSE_SECS):
        # Ignore the key press if a movement is still running.
        if not self.command_lock.acquire(blocking=False):
            return

        threading.Thread(
            target=self._pulse,
            args=(pan, tilt, zoom, secs),
            daemon=True,
        ).start()

    def _pulse(self, pan, tilt, zoom, secs):
        try:
            request = self.ptz.create_type("ContinuousMove")
            request.ProfileToken = self.token
            request.Velocity = {
                "PanTilt": {"x": pan, "y": tilt},
                "Zoom": {"x": zoom},
            }

            self.ptz.ContinuousMove(request)

            # Camera is moving from here until Stop is sent.
            t_start = time.monotonic()
            time.sleep(secs)
            t_end = time.monotonic()

            self.stop()

            # Software position = velocity x time x calibration constant.
            # Uses the MEASURED duration, not the nominal one.
            moved = (t_end - t_start) * POS_UNITS_PER_SEC
            self.state.move_by(
                dpan=pan * moved,
                dtilt=tilt * moved,
                dzoom=zoom * moved,
            )

            self.print_position()

        except Exception as e:
            # Move failed -> position is NOT changed.
            print(f"PTZ move error: {e}")

        finally:
            self.command_lock.release()

    def stop(self):
        try:
            self.ptz.Stop({
                "ProfileToken": self.token,
                "PanTilt": True,
                "Zoom": True,
            })
        except Exception as e:
            print(f"PTZ stop error: {e}")

    # ---------------- software position ----------------

    def get_state(self):
        return self.state.get()

    def print_position(self):
        pan, tilt, zoom = self.state.get()
        print(
            f"PTZ Position (software) -> "
            f"Pan: {pan:+.3f} | Tilt: {tilt:+.3f} | Zoom: {zoom:.3f}"
        )

    def reset_position(self):
        """Declare the camera's CURRENT physical pose as (0, 0, 0)."""
        self.state.reset()
        self.print_position()

    # ---------------- shutdown ----------------

    def shutdown(self):
        self.stop()


# ============================================================
# KEY MAPPING
# ============================================================

M = MOVE_SPEED

ARROW_ACTIONS = {
    # GTK
    65362: ("Tilt Up",   {"tilt": M}),
    65364: ("Tilt Down", {"tilt": -M}),
    65361: ("Pan Left",  {"pan": -M}),
    65363: ("Pan Right", {"pan": M}),

    # Qt
    16777235: ("Tilt Up",   {"tilt": M}),
    16777237: ("Tilt Down", {"tilt": -M}),
    16777234: ("Pan Left",  {"pan": -M}),
    16777236: ("Pan Right", {"pan": M}),
}

CHAR_ACTIONS = {
    "w": ("Tilt Up",   {"tilt": M}),
    "s": ("Tilt Down", {"tilt": -M}),
    "a": ("Pan Left",  {"pan": -M}),
    "d": ("Pan Right", {"pan": M}),

    "+": ("Zoom In",   {"zoom": M}),
    "=": ("Zoom In",   {"zoom": M}),
    "-": ("Zoom Out",  {"zoom": -M}),
    "_": ("Zoom Out",  {"zoom": -M}),
}


def lookup_action(key):
    if key == -1:
        return None

    # Arrow keys first (mask off NumLock bits). Their low byte collides
    # with letters, e.g. 65361 (LEFT) -> 0x51 'Q'.
    for code in (key, key & 0xFFFF):
        if code in ARROW_ACTIONS:
            return ARROW_ACTIONS[code]

    low = key & 0xFF
    if 32 <= low < 127:
        ch = chr(low).lower()
        if ch == "q":
            return ("quit", None)
        if ch == "r":
            return ("reset", None)
        return CHAR_ACTIONS.get(ch)

    return None


# ============================================================
# VIDEO OVERLAY
# ============================================================

def put_text(frame, text, x, y, scale=0.7):
    # Black outline + green text so it is readable on any background.
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 255, 0), 2, cv2.LINE_AA)


def draw_bar(frame, x, y, width, height, value, vmin, vmax, label):
    cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 255), 1)

    frac = (value - vmin) / (vmax - vmin)
    pos = int(clamp(frac, 0.0, 1.0) * width)

    cv2.line(frame, (x + pos, y), (x + pos, y + height), (0, 255, 0), 3)

    cv2.putText(frame, label, (x + width + 10, y + height - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def draw_ptz_info(frame, controller):
    pan, tilt, zoom = controller.get_state()

    put_text(frame, "PTZ (software estimate)", 20, 30, 0.6)
    put_text(frame, f"PAN  : {pan:+.3f}", 20, 62)
    put_text(frame, f"TILT : {tilt:+.3f}", 20, 94)
    put_text(frame, f"ZOOM : {zoom:.3f}", 20, 126)

    draw_bar(frame, 20, 145, 240, 15, pan, -1.0, 1.0, "PAN")
    draw_bar(frame, 20, 175, 240, 15, tilt, -1.0, 1.0, "TILT")
    draw_bar(frame, 20, 205, 240, 15, zoom, 0.0, 1.0, "ZOOM")


# ============================================================
# MAIN
# ============================================================

def main():
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

    print("Connecting to RTSP stream...")
    reader = LatestFrame(RTSP_URL).start()

    cv2.namedWindow("PTZ Camera", cv2.WINDOW_NORMAL)

    print()
    print("======================================")
    print("          PTZ CAMERA CONTROL")
    print("======================================")
    print("UP / W       -> Tilt Up")
    print("DOWN / S     -> Tilt Down")
    print("LEFT / A     -> Pan Left")
    print("RIGHT / D    -> Pan Right")
    print("+ / =        -> Zoom In")
    print("-            -> Zoom Out")
    print("R            -> Reset software position to (0,0,0)")
    print("Q            -> Quit")
    print("======================================")
    print("Position shown is a software estimate (camera gives no feedback).")
    print()

    try:
        while True:
            frame = reader.read()

            if frame is not None:
                draw_ptz_info(frame, controller)
                cv2.imshow("PTZ Camera", frame)

            # Must run every loop, even without a frame.
            key = cv2.waitKeyEx(30)

            # Window X button.
            if cv2.getWindowProperty("PTZ Camera", cv2.WND_PROP_VISIBLE) < 1:
                break

            action = lookup_action(key)
            if action is None:
                continue

            name, kwargs = action

            if name == "quit":
                break

            if name == "reset":
                controller.reset_position()
                continue

            print(f"Command: {name}")
            controller.pulse(**kwargs)

    except KeyboardInterrupt:
        pass

    finally:
        print()
        print("Stopping PTZ...")
        controller.shutdown()
        reader.stop()
        cv2.destroyAllWindows()
        print("PTZ camera control stopped.")


if __name__ == "__main__":
    main()