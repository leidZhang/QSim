import time
import pytest
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt

from core.utils.performance import plot_data_in_dict
from tests.performance_environment import prepare_test_environment
from tests.performance_environment import destroy_map
from .pid_vehicle import VisionPIDTestCar
from .constants import STEERING_DEFAULT_K_I, STEERING_DEFAULT_K_P, STEERING_DEFAULT_K_D
from .constants import THROTTLE_DEFAULT_K_P, THROTTLE_DEFAULT_K_I, THROTTLE_DEFAULT_K_D
from .constants import DEFAULT_SLOPE_OFFSET, DEFAULT_INTERCEPT_OFFSET

def run_vision_pid(expected_velocity: float, duration: float=10.3) -> Dict[str, List[float]]:
    """
    Runs the Vision PID test for a given duration and logs desired and observed velocities.

    Args:
        duration (float): The duration to run the test in seconds.

    Returns:
        Dict[str, List[float]]: A dictionary containing lists of desired and observed velocities.
    """
    history = {'desired': [], 'observed': []}

    car = VisionPIDTestCar(1, 1)
    offsets: Tuple[float, float] = (DEFAULT_SLOPE_OFFSET, DEFAULT_INTERCEPT_OFFSET)
    pid_gains = {
        'steering': [STEERING_DEFAULT_K_P, STEERING_DEFAULT_K_I, STEERING_DEFAULT_K_D],
        'throttle': [THROTTLE_DEFAULT_K_P, THROTTLE_DEFAULT_K_I, THROTTLE_DEFAULT_K_D]
    }

    car.setup(expected_velocity=expected_velocity, pid_gains=pid_gains, offsets=offsets)
    car.policy.reset_start_time()  # reset the start time
    start_time = time.time()
    try:
        while time.time() - start_time < duration:
            car.execute()
            history['desired'].append(car.policy.reference_velocity)
            history['observed'].append(car.estimate_speed())
    except KeyboardInterrupt:
        print("Interrupted!")
    finally:
        car.running_gear.read_write_std(0, 0)

    return history

@pytest.fixture(scope="function")
def my_fixture():
    assert destroy_map() != -1, "Failed to destroy the map."
    prepare_test_environment(node_id=24)  # prepare the map
    yield

def test_vision_pid(my_fixture) -> None:
    """
    Tests the Vision PID control system.
    """
    test_speed: float = 0.80
    expected_max_speed: float = 1.00
    history = run_vision_pid(expected_velocity=test_speed, duration=10.0)
    input_max_speed = round(max(history['observed']), 2)

    # plot_data_in_dict(history, title="Vision PID Test", x_label="Time (s)", y_label="Speed (m/s)")

    assert len(history['desired']) > 0, "No desired velocities logged."
    assert len(history['observed']) > 0, "No observed velocities logged."
    assert input_max_speed <= expected_max_speed, \
        f"Observed max speed {input_max_speed}m/s exceeds expected max speed {expected_max_speed}m/s."
