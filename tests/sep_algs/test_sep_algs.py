import time
from typing import List
from multiprocessing import Process, Lock, Event

import pytest

from tests.performance_environment import prepare_test_environment
from tests.performance_environment import destroy_map
from tests.vision_pid.constants import STEERING_DEFAULT_K_P, STEERING_DEFAULT_K_D, STEERING_DEFAULT_K_I
from tests.vision_pid.constants import THROTTLE_DEFAULT_K_P, THROTTLE_DEFAULT_K_D, THROTTLE_DEFAULT_K_I
from .vehicle import HardwareModule, ControlAlgModule, ObserveAlgModule

def run_hardware_process(locks: dict, desired_speed: float, duration: float = 10) -> None:
    start_time: float = time.time()
    module: HardwareModule = HardwareModule(desired_speed=desired_speed)
    module.setup(control_image_size=(410,820,3), observe_image_size=(480,640,3))
    while time.time() - start_time < duration:
        module.execute(locks=locks)
    # halt the car
    print("Halting the car...")
    module.halt_car()

def run_observe_process(lock, event, duration: float = 10) -> None:
    start_time: float = time.time()
    module: ObserveAlgModule = ObserveAlgModule(observe_image_size=(480,640,3))
    event.set()
    while time.time() - start_time < duration:
        module.execute(lock)

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

def test_sep_algs() -> None:
    prcoesses: List[Process] = []
    activate_event = Event()
    duration: float = 1000
    desired_speed: float = 1.50
    locks: dict = {
        'control': Lock(),
        'observe': Lock(),
    }
    control_process: Process = Process(
        target=run_control_process,
        args=(locks['control'], desired_speed, duration+2)
    )
    prcoesses.append(control_process)
    # observe algorithm process
    observe_process: Process = Process(
        target=run_observe_process,
        args=(locks['observe'], activate_event, duration+2)
    )
    prcoesses.append(observe_process)
    # start processes
    for process in prcoesses:
        process.start()
    activate_event.wait()
    time.sleep(2)
    run_hardware_process(locks, desired_speed, duration)