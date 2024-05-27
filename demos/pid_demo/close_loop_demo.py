import time

import numpy as np
import cv2
import matplotlib.pyplot as plt

from core.utils.tools import plot_data_in_dict
from demos.simulator_demo import simulator_demo
from .vehicle import VisionPIDTestCar
from .constants import STEERING_DEFAULT_K_I, STEERING_DEFAULT_K_P, STEERING_DEFAULT_K_D
from .constants import THROTTLE_DEFAULT_K_P, THROTTLE_DEFAULT_K_I, THROTTLE_DEFAULT_K_D

def vision_pid_demo() -> None: 
    history: dict = {
        'desired': [], 
        'observed': []
    }
    simulator_demo(node_id=24) # prepare the map
    car: VisionPIDTestCar = VisionPIDTestCar(1, 1)
    pid_gains: dict = {
        'steering': [STEERING_DEFAULT_K_P, STEERING_DEFAULT_K_I, STEERING_DEFAULT_K_D],
        'throttle': [THROTTLE_DEFAULT_K_P, THROTTLE_DEFAULT_K_I, THROTTLE_DEFAULT_K_D]
    }

    car.setup(expected_velocity=2.00, pid_gains=pid_gains)
    car.policy.reset_start_time() # reset the start time
    start_time: float = time.time()
    try: 
        while time.time() - start_time < 10.3: 
            car.execute(history=history)
    except KeyboardInterrupt:
        print("Interrupted!")
    finally:
        car.running_gear.read_write_std(0, 0)
        plot_data_in_dict(history, "Vision PID Test", "Time", "Velocity")
        