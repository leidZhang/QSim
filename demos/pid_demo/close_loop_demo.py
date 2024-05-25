import time

import numpy as np
import matplotlib.pyplot as plt

from core.utils.tools import plot_data_in_list
from demos.simulator_demo import simulator_demo
from .vehicle import VisionPIDTestCar
from .constants import STEERING_DEFAULT_K_I, STEERING_DEFAULT_K_P, STEERING_DEFAULT_K_D
from .constants import THROTTLE_DEFAULT_K_P, THROTTLE_DEFAULT_K_I, THROTTLE_DEFAULT_K_D

def vision_pid_demo() -> None: 
    history: list = []
    simulator_demo(node_id=24) # prepare the map
    car: VisionPIDTestCar = VisionPIDTestCar(1, 1)
    pid_gains: dict = {
        'steering': [STEERING_DEFAULT_K_P, STEERING_DEFAULT_K_I, STEERING_DEFAULT_K_D],
        'throttle': [THROTTLE_DEFAULT_K_P, THROTTLE_DEFAULT_K_I, THROTTLE_DEFAULT_K_D]
    }

    car.setup(expected_velocity=2.01, pid_gains=pid_gains)
    car.policy.reset_start_time() # reset the start time
    start_time: float = time.time()
    try: 
        while True: 
            car.execute(history=history)
    except KeyboardInterrupt:
        print("Interrupted!")
    finally:
        car.running_gear.read_write_std(0, 0)
        history_array: np.ndarray = np.array(history)
        print(f"\nAverage PWM: {np.mean(history_array)}, Variance: {np.var(history_array)}")
        plot_data_in_list(history, "Vision PID Test", "Time", "PWM")
        