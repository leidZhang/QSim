import time
from typing import List, Tuple
from multiprocessing import Process, Lock, Event

import pytest

from core.utils.tools import realtime_message_output
from tests.performance_environment import prepare_test_environment
from tests.performance_environment import destroy_map
from .constants import STEERING_DEFAULT_K_P, STEERING_DEFAULT_K_D, STEERING_DEFAULT_K_I
from .constants import THROTTLE_DEFAULT_K_P, THROTTLE_DEFAULT_K_D, THROTTLE_DEFAULT_K_I
from .constants import DEFAULT_SLOPE_OFFSET, DEFAULT_INTERCEPT_OFFSET
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
    # module.writer.write_images()

def run_observe_process(lock, event, duration: float = 10) -> None:
    start_time: float = time.time()
    module: ObserveAlgModule = ObserveAlgModule(observe_image_size=(480,640,3))
    event.set()
    while time.time() - start_time < duration:
        module.execute(lock)

def run_control_process(lock, desired_speed: float, duration: float = 10) -> None:
    start_time: float = time.time()
    module: ControlAlgModule = ControlAlgModule(
        control_image_size=(410,820,3), 
        will_mock_delay=True
    )
    offsets: Tuple[float, float] = (DEFAULT_SLOPE_OFFSET, DEFAULT_INTERCEPT_OFFSET)
    pid_gains = {
        'steering': [STEERING_DEFAULT_K_P, STEERING_DEFAULT_K_I, STEERING_DEFAULT_K_D],
        'throttle': [THROTTLE_DEFAULT_K_P, THROTTLE_DEFAULT_K_I, THROTTLE_DEFAULT_K_D]
    }
    module.setup(expected_velocity=desired_speed, pid_gains=pid_gains, offsets=offsets)
    while time.time() - start_time < duration:
        start: float = time.time()
        module.execute(lock)
        if time.time() - start > 0: 
            realtime_message_output(f"Frequency for iteration: {1 /(time.time() - start)}Hz")

@pytest.fixture(scope="function")
def my_fixture():
    assert destroy_map() != -1, "Failed to destroy the map."
    prepare_test_environment(node_id=24)
    yield

def test_sep_algs(my_fixture) -> None:
    prcoesses: List[Process] = []
    activate_event = Event()
    duration: float = 1000
    desired_speed: float = 1.40
    locks: dict = {
        'control': Lock(),
        'observe': Lock(),
    }
    control_process: Process = Process(
        target=run_control_process,
        name="control",
        args=(locks['control'], desired_speed, duration+2)
    )
    prcoesses.append(control_process)
    # observe algorithm process
    observe_process: Process = Process(
        target=run_observe_process,
        name="observer",
        args=(locks['observe'], activate_event, duration+2)
    )
    prcoesses.append(observe_process)
    # start processes
    for process in prcoesses:
        print(f"Activating {process.name} process...")
        process.start()
    activate_event.wait()
    time.sleep(2)
    print("Activating hardware process...")
    run_hardware_process(locks, desired_speed, duration)