import os

# Must be set BEFORE the first VideoCapture is created.
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import sys
import time
import threading

import cv2
from onvif import ONVIFCamera


# ============================================================
# CONFIGURATION
# ============================================================

IP = "192.168.1.123"
PORT = 2000
USER = "admin"
PASSWORD = "admin"

RTSP_URL = f"rtsp://{IP}:554/live/av0"

STEP_PT = 0.005        # pan/tilt change per key press (range -1.0 to 1.0)
STEP_Z = 0.005         # zoom change per key press     (range  0.0 to 1.0)
SYNC_ON_START = True  # move camera to (0,0,0) at start so software == camera


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ============================================================
# RTSP READER (keeps only the latest frame)
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
# ABSOLUTE PTZ CONTROLLER
# ============================================================
# GetStatus on this camera always returns 0, so we keep our OWN position:
# the "target" is what we last commanded. A sender thread pushes the
# latest target to the camera with AbsoluteMove. If keys are pressed
# faster than the camera can follow, intermediate targets are skipped
# and only the newest one is sent.

class AbsolutePTZ:
    def __init__(self, ptz, token):
        self.ptz = ptz
        self.token = token
        self.lock = threading.Lock()
        self.target = (0.0, 0.0, 0.0)   # pan, tilt, zoom
        self.last_sent = None
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    # ---- state changes (called from the main/UI thread) ----

    def nudge(self, dpan=0.0, dtilt=0.0, dzoom=0.0):
        with self.lock:
            p, t, z = self.target
            self.target = (
                clamp(p + dpan, -1.0, 1.0),
                clamp(t + dtilt, -1.0, 1.0),
                clamp(z + dzoom, 0.0, 1.0),
            )

    def goto(self, pan, tilt, zoom=0.0):
        with self.lock:
            self.target = (
                clamp(pan, -1.0, 1.0),
                clamp(tilt, -1.0, 1.0),
                clamp(zoom, 0.0, 1.0),
            )

    def get_target(self):
        with self.lock:
            return self.target

    # ---- sender thread ----

    def _run(self):
        while self.running:
            with self.lock:
                snap = self.target
            if snap != self.last_sent:
                self._send(*snap)
                self.last_sent = snap
                print(f"Pan: {snap[0]:+.3f} | Tilt: {snap[1]:+.3f} | Zoom: {snap[2]:.3f}")
            time.sleep(0.05)

    def _send(self, pan, tilt, zoom):
        try:
            req = self.ptz.create_type("AbsoluteMove")
            req.ProfileToken = self.token
            req.Position = {
                "PanTilt": {"x": pan, "y": tilt},
                "Zoom": {"x": zoom},
            }
            self.ptz.AbsoluteMove(req)
        except Exception as e:
            print(f"AbsoluteMove error: {e}")

    def stop(self):
        self.running = False
        self.thread.join(timeout=2)
        try:
            self.ptz.Stop({
                "ProfileToken": self.token,
                "PanTilt": True,
                "Zoom": True,
            })
        except Exception as e:
            print(f"PTZ stop error: {e}")


# ============================================================
# TEST SWEEP (checks whether the camera really obeys AbsoluteMove)
# ============================================================

def run_test_sweep(controller):
    def worker():
        print("--- TEST SWEEP: watch the camera physically ---")
        steps = [
            (-0.5, 0.0, 0.0), (0.5, 0.0, 0.0),
            (0.0, 0.5, 0.0), (0.0, -0.5, 0.0),
            (0.0, 0.0, 0.0),
        ]
        for p, t, z in steps:
            controller.goto(p, t, z)
            time.sleep(3)
        print("--- TEST SWEEP DONE ---")
    threading.Thread(target=worker, daemon=True).start()


# ============================================================
# KEY MAPPING
# ============================================================
# Arrow codes: GTK 65361-65364, Qt 16777234-16777237. WASD as fallback.

ARROW_ACTIONS = {
    65362: ("up", None), 65364: ("down", None),
    65361: ("left", None), 65363: ("right", None),
    16777235: ("up", None), 16777237: ("down", None),
    16777234: ("left", None), 16777236: ("right", None),
}

CHAR_ACTIONS = {
    "w": "up", "s": "down", "a": "left", "d": "right",
    "+": "zoom_in", "=": "zoom_in",
    "-": "zoom_out", "_": "zoom_out",
    "o": "center", "t": "test", "q": "quit",
}


def lookup_action(key):
    if key == -1:
        return None
    # Arrow codes first (their low byte collides with letters like 'Q').
    for code in (key, key & 0xFFFF):
        if code in ARROW_ACTIONS:
            return ARROW_ACTIONS[code][0]
    low = key & 0xFF
    if 32 <= low < 127:
        return CHAR_ACTIONS.get(chr(low).lower())
    return None


# ============================================================
# MAIN
# ============================================================

def main():
    print("Connecting to PTZ camera...")
    try:
        cam = ONVIFCamera(IP, PORT, USER, PASSWORD)
        media = cam.create_media_service()
        ptz = cam.create_ptz_service()
        token = media.GetProfiles()[0].token
    except Exception as e:
        print(f"ERROR: ONVIF connection failed: {e}")
        sys.exit(1)
    print("ONVIF connected.")

    controller = AbsolutePTZ(ptz, token)
    if SYNC_ON_START:
        controller.goto(0.0, 0.0, 0.0)   # sender thread moves camera to 0,0,0

    reader = LatestFrame(RTSP_URL).start()
    cv2.namedWindow("PTZ Camera", cv2.WINDOW_NORMAL)

    print()
    print("===================================")
    print("      PTZ ABSOLUTE CONTROL")
    print("===================================")
    print("UP / W      -> Tilt up")
    print("DOWN / S    -> Tilt down")
    print("LEFT / A    -> Pan left")
    print("RIGHT / D   -> Pan right")
    print("+ / =       -> Zoom in")
    print("-           -> Zoom out")
    print("O           -> Go to centre (0, 0, 0)")
    print("T           -> Test sweep (watch the camera)")
    print("Q           -> Quit")
    print("===================================")
    print("Position shown = commanded position (camera gives no feedback)")

    try:
        while True:
            frame = reader.read()
            if frame is not None:
                p, t, z = controller.get_target()
                text = f"Pan {p:+.2f}  Tilt {t:+.2f}  Zoom {z:.2f}"
                cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow("PTZ Camera", frame)

            key = cv2.waitKeyEx(30)

            if cv2.getWindowProperty("PTZ Camera", cv2.WND_PROP_VISIBLE) < 1:
                break

            action = lookup_action(key)
            if action is None:
                continue

            if action == "quit":
                break
            elif action == "up":
                controller.nudge(dtilt=STEP_PT)
            elif action == "down":
                controller.nudge(dtilt=-STEP_PT)
            elif action == "left":
                controller.nudge(dpan=-STEP_PT)
            elif action == "right":
                controller.nudge(dpan=STEP_PT)
            elif action == "zoom_in":
                controller.nudge(dzoom=STEP_Z)
            elif action == "zoom_out":
                controller.nudge(dzoom=-STEP_Z)
            elif action == "center":
                controller.goto(0.0, 0.0, 0.0)
            elif action == "test":
                run_test_sweep(controller)

    except KeyboardInterrupt:
        pass
    finally:
        controller.stop()
        reader.stop()
        cv2.destroyAllWindows()
        print("PTZ absolute control stopped.")


if __name__ == "__main__":
    main()