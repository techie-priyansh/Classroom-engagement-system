import cv2


class Camera:
    def __init__(self, source=0):
        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise RuntimeError("Unable to open camera.")

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()