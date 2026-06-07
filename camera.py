import cv2
import threading
from config import CAMERA_SOURCE, CAPTURE_WIDTH, CAPTURE_HEIGHT, TARGET_FPS

class CameraStream:
    def __init__(self):
        self.stream = cv2.VideoCapture(CAMERA_SOURCE)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
        self.stream.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            self.ret, self.frame = self.stream.read()

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()