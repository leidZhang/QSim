import time

import cv2
import numpy as np

from core.qcar import VirtualCSICamera, VirtualRGBDCamera
from .vehicle import TestBed, PhysicalCar

def run_rgbd_thread(duration: float = 20) -> None:
    camera: VirtualRGBDCamera = VirtualRGBDCamera()
    start: float = time.time()
    while time.time() - start < duration:
        image: np.ndarray = camera.read_rgb_image()
        if image is not None:
            cv2.imshow('RGB Image', image)
            cv2.waitKey(1)

def run_csi_thread(duration: float = 20) -> None:
    camera: VirtualCSICamera = VirtualCSICamera(id=3)
    start: float = time.time()
    while time.time() - start < duration:
        image: np.ndarray = camera.read_image()
        if image is not None:
            cv2.imshow('CSI Image', image)
            cv2.waitKey(1)

def run_hardware_process(duration: float = 20) -> None:
    car: PhysicalCar = TestBed(throttle_coeff=1, steering_coeff=1)
    start: float = time.time()
    last_reset: float = start
    reverse_flag: bool = False
    while time.time() - start < duration:
        if time.time() - last_reset > 5:
            last_reset = time.time()
            reverse_flag = not reverse_flag
        car.execute(0.5 * (-1 if reverse_flag else 1))