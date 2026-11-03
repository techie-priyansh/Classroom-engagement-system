import os

# Must be set BEFORE the first VideoCapture is created.
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import sys
import json
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
# SETTINGS  (edit these)
# ============================================================

# "teach" = drive the camera by keyboard and SAVE the 4 positions.
# "sweep" = automatically visit the saved positions.
# A command-line argument overrides this:  python ptz_preset_sweep.py sweep
MODE = "sweep"
#MODE = "teach"
WAIT_SECS = 5.0          # hold time at each region (sweep mode)

# The camera does not report when it has arrived, so after every GotoPreset
# we wait this long for the move to finish BEFORE the hold timer starts.
# Time your slowest move (e.g. Region 3 -> Reset) and add a second or two.
TRAVEL_SECS = 5.0

START_FROM_RESET = True  # sweep: first go to the saved Reset position
SHOW_VIDEO = True        # False = no window (sweep mode only)

# (velocity, pulse seconds) used by the arrow keys in teach mode.
COARSE = (0.8, 0.20)
FINE = (0.3, 0.10)       # press F to switch; for precise positioning

# Where the preset tokens are remembered (next to this script).
PRESET_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "ptz_presets.json"
)

# role -> (label shown on screen, preset name stored in the camera)
ROLES = {
    "reset":   ("Reset",             "KGP_RESET"),
    "region1": ("Region 1 (right)",  "KGP_REGION1"),
    "region2": ("Region 2 (middle)", "KGP_REGION2"),
    "region3": ("Region 3 (left)",   "KGP_REGION3"),
}


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
# PTZ CONTROLLER (keyboard pulses + camera presets)
# ============================================================
# Positions are stored INSIDE the camera as presets, so they are real,
# repeatable positions. Nothing here estimates a position from timing.

class PTZController:

    def __init__(self, ptz, token):
        self.ptz = ptz
        self.token = token

        self.tokens = self.load_tokens()   # role -> preset token
        self.busy = threading.Lock()       # one teach-mode command at a time
        self.abort = threading.Event()
        self.status = "Ready"
        self.fine = False

    # ---------------- remembered preset tokens ----------------

    @staticmethod
    def load_tokens():
        try:
            with open(PRESET_FILE) as f:
                data = json.load(f)
            return {k: str(v) for k, v in data.items() if k in ROLES}
        except (OSError, ValueError):
            return {}

    def _store_tokens(self):
        with open(PRESET_FILE, "w") as f:
            json.dump(self.tokens, f, indent=2)

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
        """One short ContinuousMove. dp/dt/dz are -1, 0 or +1."""
        speed, secs = FINE if self.fine else COARSE

        request = self.ptz.create_type("ContinuousMove")
        request.ProfileToken = self.token
        request.Velocity = {
            "PanTilt": {"x": dp * speed, "y": dt * speed},
            "Zoom": {"x": dz * speed},
        }

        try:
            self.ptz.ContinuousMove(request)
            time.sleep(secs)
        finally:
            self.stop()   # always stop, even if something went wrong

    # ---------------- presets ----------------

    def _preset_tokens_on_camera(self):
        presets = self.ptz.GetPresets({"ProfileToken": self.token}) or []
        return {str(p.token) for p in presets}

    def save_preset(self, role):
        """Store the camera's CURRENT position as this role's preset."""
        label, name = ROLES[role]

        before = self._preset_tokens_on_camera()

        request = self.ptz.create_type("SetPreset")
        request.ProfileToken = self.token
        request.PresetName = name

        # Saving this role again overwrites its own preset, not a new one.
        existing = self.tokens.get(role)
        if existing and existing in before:
            request.PresetToken = existing

        response = self.ptz.SetPreset(request)

        # Most cameras return the token; some return nothing.
        token = getattr(response, "PresetToken", response)
        token = str(token) if token else None

        if not token:
            new = self._preset_tokens_on_camera() - before
            if len(new) == 1:
                token = new.pop()
            elif existing and existing in before:
                token = existing

        if not token:
            raise RuntimeError("Camera did not confirm the preset was saved.")

        self.tokens[role] = token
        self._store_tokens()

        self.status = f"Saved {label}"
        print(f"Saved {label} as camera preset '{token}'")

    def goto_preset(self, role):
        """Tell the camera to move to this role's saved position."""
        label = ROLES[role][0]

        token = self.tokens.get(role)
        if token is None:
            raise RuntimeError(
                f"{label} has not been saved yet. Run teach mode first."
            )

        request = self.ptz.create_type("GotoPreset")
        request.ProfileToken = self.token
        request.PresetToken = token
        self.ptz.GotoPreset(request)

        print(f"Going to {label} (preset '{token}')")

    # ---------------- teach-mode commands (non-blocking) ----------------

    def submit(self, fn, *args):
        """Run fn in a thread. Ignored if another command is still running."""
        if not self.busy.acquire(blocking=False):
            return False

        def runner():
            try:
                fn(*args)
            except Exception as e:
                print(f"PTZ error: {e}")
                self.status = f"Error: {e}"
            finally:
                self.busy.release()

        threading.Thread(target=runner, daemon=True).start()
        return True

    # ---------------- waiting ----------------

    def sleep(self, secs, label):
        """Wait with an on-screen countdown. Returns False if aborted."""
        end = time.monotonic() + secs
        while True:
            left = end - time.monotonic()
            if left <= 0:
                return True
            self.status = f"{label} - {left:.1f}s"
            if self.abort.wait(min(0.1, left)):
                return False


# ============================================================
# SWEEP
# ============================================================

def run_sweep(c):
    """Reset -> Region 1 -> Region 2 -> Region 3, holding at each region."""
    try:
        steps = []
        if START_FROM_RESET:
            steps.append(("reset", False))
        steps += [("region1", True), ("region2", True), ("region3", True)]

        for role, hold in steps:
            label = ROLES[role][0]

            c.status = f"Moving to {label}"
            print(f"\n--- {label} ---")
            c.goto_preset(role)

            if not c.sleep(TRAVEL_SECS, f"{label}: moving"):
                return

            if hold and not c.sleep(WAIT_SECS, f"{label}: holding"):
                return

        c.status = "Sweep complete"
        print("\nSweep complete.")

    except Exception as e:
        print(f"\nSweep stopped by error: {e}")
        c.abort.set()
        c.status = f"Error: {e}"

    finally:
        c.stop()


# ============================================================
# KEY MAPPING (teach mode)
# ============================================================
# Arrow codes: GTK 65361-65364, Qt 16777234-16777237. WASD also works.

ARROWS = {
    65362: (0, 1, 0),   65364: (0, -1, 0),     # GTK up / down
    65361: (-1, 0, 0),  65363: (1, 0, 0),      # GTK left / right
    16777235: (0, 1, 0),  16777237: (0, -1, 0),
    16777234: (-1, 0, 0), 16777236: (1, 0, 0),
}

CHARS = {
    "w": ("move", (0, 1, 0)),  "s": ("move", (0, -1, 0)),
    "a": ("move", (-1, 0, 0)), "d": ("move", (1, 0, 0)),
    "+": ("move", (0, 0, 1)),  "=": ("move", (0, 0, 1)),
    "-": ("move", (0, 0, -1)), "_": ("move", (0, 0, -1)),

    "0": ("save", "reset"),   "1": ("save", "region1"),
    "2": ("save", "region2"), "3": ("save", "region3"),

    "z": ("goto", "reset"),   "x": ("goto", "region1"),
    "c": ("goto", "region2"), "v": ("goto", "region3"),

    "f": ("fine", None),
    "q": ("quit", None),
}


def lookup_action(key):
    if key == -1:
        return None

    # Arrow keys first: their low byte looks like a letter (65361 -> 'Q').
    for code in (key, key & 0xFFFF):
        if code in ARROWS:
            return ("move", ARROWS[code])

    low = key & 0xFF
    if 32 <= low < 127:
        return CHARS.get(chr(low).lower())
    return None


# ============================================================
# VIDEO OVERLAY
# ============================================================

def put_text(frame, text, x, y, scale=0.6):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 255, 0), 2, cv2.LINE_AA)


def draw_overlay(frame, c, mode):
    y = 30
    put_text(frame, f"PTZ PRESETS - {mode.upper()} MODE", 20, y, 0.7)
    y += 32
    put_text(frame, c.status, 20, y, 0.65)
    y += 32

    if mode == "teach":
        put_text(frame, f"Speed: {'FINE' if c.fine else 'COARSE'}  (F to switch)", 20, y)
        y += 28
        put_text(frame, "Arrows/WASD move   +/- zoom", 20, y)
        y += 28
        put_text(frame, "Save here as:  0 Reset  1 Region1  2 Region2  3 Region3", 20, y)
        y += 28
        put_text(frame, "Go to saved:   Z Reset  X Region1  C Region2  V Region3", 20, y)
        y += 28
        done = [ROLES[r][0] for r in ROLES if r in c.tokens]
        put_text(frame, "Saved: " + (", ".join(done) if done else "nothing yet"), 20, y)
        y += 28
        put_text(frame, "Q quit", 20, y)
    else:
        put_text(frame, "Q = abort", 20, y)


# ============================================================
# MAIN
# ============================================================

def connect():
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
    return PTZController(ptz, token)


def is_quit(key):
    # Full-code compare so arrow keys can never count as 'Q'.
    return (key & 0xFFFF) in (ord("q"), ord("Q"))


def window_closed():
    return cv2.getWindowProperty("PTZ Camera", cv2.WND_PROP_VISIBLE) < 1


def run_teach(c, reader):
    print()
    print("======================================")
    print("        TEACH MODE - SAVE PRESETS")
    print("======================================")
    print("1. Move the camera to a position with the arrow keys.")
    print("2. Press 0 / 1 / 2 / 3 to save it as Reset / Region1 / Region2 / Region3.")
    print("   (Region 2 and 3: only PAN between saves so tilt and zoom stay equal.)")
    print("3. Press Z / X / C / V to check that the camera returns there.")
    print("F = fine/coarse speed      Q = quit")
    print("======================================")

    while True:
        frame = reader.read()
        if frame is not None:
            draw_overlay(frame, c, "teach")
            cv2.imshow("PTZ Camera", frame)

        key = cv2.waitKeyEx(30)

        if window_closed():
            break

        action = lookup_action(key)
        if action is None:
            continue

        kind, arg = action

        if kind == "quit":
            break
        elif kind == "fine":
            c.fine = not c.fine
            c.status = f"{'Fine' if c.fine else 'Coarse'} speed"
        elif kind == "move":
            c.submit(c.pulse, *arg)
        elif kind == "save":
            c.submit(c.save_preset, arg)
        elif kind == "goto":
            c.status = f"Going to {ROLES[arg][0]}"
            c.submit(c.goto_preset, arg)


def run_sweep_with_video(c, reader):
    sweep = threading.Thread(target=run_sweep, args=(c,), daemon=True)
    sweep.start()

    while sweep.is_alive():
        frame = reader.read()
        if frame is not None:
            draw_overlay(frame, c, "sweep")
            cv2.imshow("PTZ Camera", frame)

        key = cv2.waitKeyEx(30)

        if is_quit(key) or window_closed():
            print("Aborting.")
            c.abort.set()
            break

    sweep.join(timeout=5)


def main():
    mode = (sys.argv[1] if len(sys.argv) > 1 else MODE).strip().lower()
    if mode not in ("teach", "sweep"):
        sys.exit("Usage: python ptz_preset_sweep.py [teach|sweep]")

    controller = connect()

    if mode == "sweep":
        needed = ["region1", "region2", "region3"]
        if START_FROM_RESET:
            needed.insert(0, "reset")
        missing = [ROLES[r][0] for r in needed if r not in controller.tokens]
        if missing:
            sys.exit("Missing saved presets: " + ", ".join(missing)
                     + "\nRun in teach mode first and save them.")

    reader = None

    try:
        if mode == "teach" or SHOW_VIDEO:
            reader = LatestFrame(RTSP_URL).start()
            cv2.namedWindow("PTZ Camera", cv2.WINDOW_NORMAL)

        if mode == "teach":
            run_teach(controller, reader)
        elif SHOW_VIDEO:
            run_sweep_with_video(controller, reader)
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
        print("Script finished.")


if __name__ == "__main__":
    main()