import cv2
from threading import Thread
import time
# FIXED: Removed '.' before config
from config import CAMERA_SOURCE

class VisionStream:
    def __init__(self):
        self.src = CAMERA_SOURCE
        self.cap = cv2.VideoCapture(self.src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Minimize internal buffer
        
        # Threading state
        self.stopped = False
        self.grabbed = False
        self.frame = None
        
        # Check connection
        if not self.cap.isOpened():
            print(f"[Vision] Warning: Could not open {self.src}. Retrying...")

    def start(self):
        """Starts the thread to read frames from the video stream."""
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        """Keep looping infinitely until the thread is stopped."""
        while True:
            if self.stopped:
                self.cap.release()
                return

            # Grab the frame but don't fully decode unless asked (latency trick)
            grabbed, frame = self.cap.read()
            
            if grabbed:
                self.grabbed = grabbed
                self.frame = frame
            else:
                # If stream disconnects, try to reconnect briefly
                time.sleep(0.1)

    def read(self):
        """Return the most recent frame."""
        return self.frame

    def stop(self):
        self.stopped = True