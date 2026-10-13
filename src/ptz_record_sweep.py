"""Record a video while the camera runs the saved-preset sweep.

Reuses ptz_preset_sweep.py (keep both files in the same folder), so its
settings (IP, TRAVEL_SECS, WAIT_SECS, START_FROM_RESET, ...) and the presets
you saved in teach mode are used as they are.

Run:  python ptz_record_sweep.py
"""

import os
import sys
import time
import threading
from datetime import datetime

import cv2

import ptz_preset_sweep as base


# ============================================================
# SETTINGS  (edit these)
# ============================================================

SAVE_DIR = "/home/mylinux/Desktop/MTech_Project/MTP/Classroom-engagement-system/dataset"

RECORD_FPS = 25.0   # frame rate of the saved video
PAD_SECS = 1.0      # extra seconds recorded before the sweep starts and after it ends
PREVIEW = True      # show a live window while recording (the SAVED video never has the overlay)

# True  = go to Reset once, then repeat Region 1 -> 2 -> 3 until you press Q.
# False = run the sweep just once (like before) and stop by itself.
LOOP = True

# Travel time for the long jump Region 3 -> Region 1 at the end of each round.
# None = use TRAVEL_SECS from ptz_preset_sweep.py. Set a bigger number if the
# camera has not finished moving when the Region 1 hold starts.
WRAP_TRAVEL_SECS = None


# ============================================================
# RECORDER
# ============================================================
# The camera stream delivers frames at irregular times, so a separate thread
# writes the newest frame at a fixed rate. If the stream stalls, the last
# frame is repeated, so the video length always equals the real time elapsed.
# Frames are written exactly as received: no text or overlay is drawn on them.

class Recorder:

    def __init__(self, reader, path, fps):
        self.reader = reader
        self.path = path
        self.fps = fps
        self.writer = None
        self.size = None            # (width, height)
        self.frames = 0
        self.running = False
        self.thread = None

    def start(self, timeout=15.0):
        # Wait for the first frame: it tells us the video size.
        first = None
        end = time.monotonic() + timeout
        while first is None and time.monotonic() < end:
            first = self.reader.read()
            if first is None:
                time.sleep(0.05)

        if first is None:
            raise RuntimeError("No video frames received from the camera.")

        height, width = first.shape[:2]
        self.size = (width, height)

        # mp4v works with the standard OpenCV pip build. (H.264 usually needs ffmpeg.)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(self.path, fourcc, self.fps, self.size)
        if not self.writer.isOpened():
            raise RuntimeError(f"Could not open video file for writing: {self.path}")

        self._first = first
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        period = 1.0 / self.fps
        last = self._first
        next_t = time.monotonic()

        while self.running:
            now = time.monotonic()

            if now < next_t:
                time.sleep(min(0.005, next_t - now))
                continue

            frame = self.reader.read()
            # Ignore a frame of a different size (e.g. after a stream reconnect).
            if frame is not None and (frame.shape[1], frame.shape[0]) == self.size:
                last = frame

            self.writer.write(last)
            self.frames += 1
            next_t += period

            # After a long stall, resync instead of writing a burst of copies.
            if now - next_t > 2.0:
                next_t = now

    def elapsed(self):
        return self.frames / self.fps

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=3)
        if self.writer is not None:
            self.writer.release()    # finalises the file; without it the mp4 is broken
            self.writer = None
        return self.frames


# ============================================================
# RECORD + SWEEP
# ============================================================

def run_loop(c):
    """Reset ONCE, then Region 1 -> 2 -> 3 -> 1 -> 2 -> 3 ... until abort (Q)."""
    c.rounds_done = 0
    try:
        if base.START_FROM_RESET:
            c.status = "Moving to Reset"
            print("\n--- Reset (only once) ---")
            c.goto_preset("reset")
            if not c.sleep(base.TRAVEL_SECS, "Reset: moving"):
                return

        round_no = 0
        while not c.abort.is_set():
            round_no += 1
            print(f"\n===== Round {round_no} =====")

            for role in ("region1", "region2", "region3"):
                label = base.ROLES[role][0]

                # Region 1 after the first round follows Region 3 (long jump back).
                travel = base.TRAVEL_SECS
                if role == "region1" and round_no > 1 and WRAP_TRAVEL_SECS is not None:
                    travel = WRAP_TRAVEL_SECS

                print(f"--- {label} ---")
                c.goto_preset(role)

                if not c.sleep(travel, f"Round {round_no} {label}: moving"):
                    return
                if not c.sleep(base.WAIT_SECS, f"Round {round_no} {label}: holding"):
                    return

            c.rounds_done = round_no

    except Exception as e:
        print(f"\nLoop stopped by error: {e}")
        c.abort.set()
        c.status = f"Error: {e}"

    finally:
        c.stop()   # never leave the camera moving


def draw_rec_indicator(frame, recorder):
    secs = int(recorder.elapsed())
    text = f"REC {secs // 60:02d}:{secs % 60:02d}"
    x = frame.shape[1] - 190
    cv2.circle(frame, (x, 30), 9, (0, 0, 255), -1)
    cv2.putText(frame, text, (x + 18, 38), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (x + 18, 38), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 0, 255), 2, cv2.LINE_AA)


def record_sweep(controller, reader, output_file, preview=True):
    """Record while the preset sweep runs. Returns the number of frames written."""
    recorder = Recorder(reader, output_file, RECORD_FPS)
    recorder.start()
    print(f"Recording to: {output_file}")

    def job():
        # Pad, sweep, pad. An abort (Q / window closed / error) skips the rest.
        if controller.abort.wait(PAD_SECS):
            return
        if LOOP:
            run_loop(controller)          # returns only when Q / error
        else:
            base.run_sweep(controller)    # one pass, then stops
        if not controller.abort.is_set():
            controller.status = "Finishing recording..."
            controller.abort.wait(PAD_SECS)

    worker = threading.Thread(target=job, daemon=True)

    try:
        if preview:
            cv2.namedWindow("PTZ Camera", cv2.WINDOW_NORMAL)

        worker.start()

        while worker.is_alive():
            if preview:
                frame = reader.read()
                if frame is not None:
                    # Overlay is drawn on this display copy ONLY.
                    base.draw_overlay(frame, controller, "sweep")
                    draw_rec_indicator(frame, recorder)
                    cv2.imshow("PTZ Camera", frame)

                key = cv2.waitKeyEx(30)

                if base.is_quit(key) or base.window_closed():
                    print("Stopping - the video recorded so far is saved.")
                    controller.abort.set()
                    break
            else:
                worker.join(0.2)

        worker.join(timeout=10)

    except KeyboardInterrupt:
        print("Ctrl+C - aborting (the video recorded so far is kept).")
        controller.abort.set()

    finally:
        controller.stop()
        frames = recorder.stop()
        if preview:
            cv2.destroyAllWindows()

    return frames


# ============================================================
# MAIN
# ============================================================

def main():
    controller = base.connect()

    # Every preset the sweep needs must already be saved (teach mode).
    needed = ["region1", "region2", "region3"]
    if base.START_FROM_RESET:
        needed.insert(0, "reset")
    missing = [base.ROLES[r][0] for r in needed if r not in controller.tokens]
    if missing:
        sys.exit("Missing saved presets: " + ", ".join(missing)
                 + "\nRun ptz_preset_sweep.py in teach mode first and save them.")

    os.makedirs(SAVE_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(SAVE_DIR, f"ptz_recording_{timestamp}.mp4")

    reader = base.LatestFrame(base.RTSP_URL).start()

    try:
        frames = record_sweep(controller, reader, output_file, preview=PREVIEW)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        controller.stop()
        reader.stop()
        sys.exit(1)

    reader.stop()

    print()
    if frames > 0 and os.path.exists(output_file):
        size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"Saved: {output_file}")
        print(f"       {frames} frames, {frames / RECORD_FPS:.1f} s, {size_mb:.1f} MB")
        if LOOP:
            print(f"       {getattr(controller, 'rounds_done', 0)} full rounds of Region 1-2-3 completed")
    else:
        print("WARNING: no video was saved.")


if __name__ == "__main__":
    main()