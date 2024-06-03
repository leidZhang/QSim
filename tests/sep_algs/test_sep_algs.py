import time
from multiprocessing import Process, Lock

import pytest

from tests.performance_environment import prepare_test_environment
from tests.performance_environment import destroy_map
from tests.vision_pid.constants import STEERING_DEFAULT_K_P, STEERING_DEFAULT_K_D, STEERING_DEFAULT_K_I
from tests.vision_pid.constants import THROTTLE_DEFAULT_K_P, THROTTLE_DEFAULT_K_D, THROTTLE_DEFAULT_K_I
from .vehicle import HardwareModule, ControlAlgModule, ObserveAlgModule

def run_hardware_process(locks: dict, desired_speed: float, duration: float = 10) -> None:
    try:
        start_time: float = time.time()
        module: HardwareModule = HardwareModule(desired_speed=desired_speed)
        module.setup(control_image_size=(410,820,3), observe_image_size=(410,820,3))
        while time.time() - start_time < duration:
            module.execute(locks=locks)
    finally:
        module.halt_car()

def run_observe_process(lock, duration: float = 10) -> None:
    module: ObserveAlgModule = ObserveAlgModule(observe_image_size=(410,820,3))

def run_control_process(lock, desired_speed: float, duration: float = 10) -> None:
    start_time: float = time.time()
    module: ControlAlgModule = ControlAlgModule(control_image_size=(410,820,3))
    pid_gains = {
        'steering': [STEERING_DEFAULT_K_P, STEERING_DEFAULT_K_I, STEERING_DEFAULT_K_D],
        'throttle': [THROTTLE_DEFAULT_K_P, THROTTLE_DEFAULT_K_I, THROTTLE_DEFAULT_K_D]
    }
    module.setup(expected_velocity=desired_speed, pid_gains=pid_gains)
    while time.time() - start_time < duration:
        module.execute(lock)

@pytest.fixture(scope="function")
def my_fixture():
    assert destroy_map() != -1, "Failed to destroy the map."
    prepare_test_environment(node_id=24)
    yield

def test_sep_algs(my_fixture) -> None:
    prcoesses: list = []
    duration: float = 19
    desired_speed: float = 1.00
    locks: dict = {
        'control': Lock(),
        'observe': Lock(),
    }
    # control algorithm process
    control_process: Process = Process(
        target=run_control_process,
        args=(locks['control'], desired_speed, duration+2)
    )
    prcoesses.append(control_process)
    # observe algorithm process
    observe_process: Process = Process(
        target=run_observe_process,
        args=(locks['observe'], duration+2)
    )
    prcoesses.append(observe_process)
    # start processes
    for process in prcoesses:
        process.start()
    time.sleep(2)
    run_hardware_process(locks, desired_speed, duration)