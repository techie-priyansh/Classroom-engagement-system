import cv2


class Camera:
    def __init__(self, source="rtsp://192.168.1.123:554/live/av0"):
        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise RuntimeError("Unable to open camera.")

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()
