
from paths import DATASET_DIR
import cv2
import os
from datetime import datetime


# =========================
# Camera Configuration
# =========================

RTSP_URL = "rtsp://192.168.1.123:554/live/av0"

SAVE_DIR = str(DATASET_DIR)


# =========================
# Create Dataset Directory
# =========================

os.makedirs(SAVE_DIR, exist_ok=True)


# =========================
# Connect to Camera
# =========================

cap = cv2.VideoCapture(RTSP_URL)

if not cap.isOpened():
    print("ERROR: Could not connect to PTZ camera.")
    exit()

print("PTZ camera connected successfully.")


# =========================
# Get Video Properties
# =========================

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps <= 0:
    fps = 25.0


# =========================
# Create Recording File
# =========================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

output_file = os.path.join(
    SAVE_DIR,
    f"ptz_recording_{timestamp}.mp4"
)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    output_file,
    fourcc,
    fps,
    (width, height)
)


# =========================
# Real-Time Recording
# =========================

print(f"Resolution : {width} x {height}")
print(f"FPS        : {fps:.2f}")
print(f"Saving to  : {output_file}")
print("Recording started.")
print("Press 'q' to stop recording.")


while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Failed to receive frame from camera.")
        break

    # Display live video
    cv2.imshow("PTZ Camera - Live", frame)

    # Save frame
    writer.write(frame)

    # Stop with Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =========================
# Cleanup
# =========================

cap.release()
writer.release()
cv2.destroyAllWindows()

print("Recording stopped.")
print(f"Video saved at: {output_file}")
