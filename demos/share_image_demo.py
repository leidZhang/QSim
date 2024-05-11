import time
from multiprocessing import Process, Lock, Event
from multiprocessing.shared_memory import SharedMemory

import cv2
import numpy as np

def run_camera_process(shm_name, lock, frame_shape, frame_dtype):
    cap = cv2.VideoCapture(0)
    shm = SharedMemory(name=shm_name)
    img_shared = np.ndarray(frame_shape, dtype=frame_dtype, buffer=shm.buf)
    while True:
        _, img = cap.read()
        img = cv2.flip(img, 1)
        with lock:
            np.copyto(img_shared, img)

def run_receive_process(shm_name, lock, frame_shape, frame_dtype):
    shm = SharedMemory(name=shm_name)
    img_shared = np.ndarray(frame_shape, dtype=frame_dtype, buffer=shm.buf)
    while True:
        with lock:
            img = np.copy(img_shared)
        cv2.imshow("Video Frame", img)
        grey_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(grey_image, (5, 5), 0)
        edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
        cv2.imshow('Edges', edges)
        cv2.waitKey(1)

def get_img_size():
    cap = cv2.VideoCapture(0)
    _, frame = cap.read()
    cap.release()

    shape = frame.shape
    dtype = frame.dtype
    nbytes = frame.nbytes

    return shape, dtype, nbytes

def start_share_image(shm_name):
    lock = Lock()
    dtype = np.uint8
    shape = (480, 640, 3)
    nbytes = 921600

    shm = SharedMemory(name=shm_name, create=True, size=nbytes)
    p1 = Process(target=run_camera_process, args=(shm.name, lock, shape, dtype))
    p2 = Process(target=run_receive_process, args=(shm.name, lock, shape, dtype))
    p1.start()
    p2.start()
    while True:
        time.sleep(100)

if __name__ == "__main__":
    shm_name = "notebook_cap"
    start_share_image(shm_name)
