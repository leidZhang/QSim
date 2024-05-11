import os
import time
from multiprocessing import Lock, Process
from multiprocessing.shared_memory import SharedMemory

import numpy as np

from qvl.qlabs import QuanserInteractiveLabs

from core.roadmap import ACCDirector
from demos.override_demo import prepare_map_info
from .car import PPCar, PPCarMP

def generate_car(actor_info) -> None:
    actor_info[0].spawn_id(
        actorNumber=0,
        location=actor_info[1],
        rotation=actor_info[2],
        scale=[.1, .1, .1],
        configuration=0,
        waitForConfirmation=True
    )

def run_pure_pursuite() -> None:
    qlabs: QuanserInteractiveLabs = QuanserInteractiveLabs()
    qlabs.open("localhost")
    director: ACCDirector = ACCDirector(qlabs)

    init_pos, waypoint_sequence = prepare_map_info(node_id=24)
    actors: dict = director.build_map(init_pos)
    generate_car(actors["car"])
    car: PPCar = PPCar(qlabs=qlabs, waypoint_sequence=waypoint_sequence)
    car.setup()

    done = False
    while not done:
        try:
            start = time.time()
            car.execute()
            # print(f"Run with {1 / (time.time() - start)} Hz")
        except IndexError:
            print("Reached the goal!")
            car.halt_car()
            done = True

    qlabs.close()

def run_subject(shm_name: str, lock) -> None:
    qlabs: QuanserInteractiveLabs = QuanserInteractiveLabs()
    qlabs.open("localhost")
    director: ACCDirector = ACCDirector(qlabs)

    init_pos, waypoint_sequence = prepare_map_info(node_id=24)
    actors: dict = director.build_map(init_pos)
    generate_car(actors["car"])
    car: PPCarMP = PPCarMP(qlabs=qlabs, waypoint_sequence=waypoint_sequence)
    car.setup()

    done = False
    while not done:
        try:
            car.execute(shm_name, lock)
        except IndexError:
            print("Reached the goal!")
            car.halt_car()
            done = True

    qlabs.close()

def run_observer(shm_name: str, lock) -> None:
    shm = SharedMemory(name=shm_name)
    action_shared: np.ndarray = np.ndarray((2,), dtype=np.float64, buffer=shm.buf)
    while True:
        with lock:
            action: np.ndarray = action_shared.copy()
        print(action)

def start_test_mp() -> None:
    lock = Lock()
    shm_name = "test"
    shm = SharedMemory(name=shm_name, create=True, size=16)
    p1 = Process(target=run_subject, args=(shm_name, lock))
    p2 = Process(target=run_observer, args=(shm_name, lock))
    p1.start()
    p2.start()
    while True:
        time.sleep(100)