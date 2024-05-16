import sys
import time
import os
from queue import Queue
from typing import List
from multiprocessing import Lock, Process, Event
from multiprocessing.shared_memory import SharedMemory

import cv2
import numpy as np

from qvl.real_time import QLabsRealTime
from qvl.qlabs import QuanserInteractiveLabs
from qvl.qcar import QLabsQCar
import pal.resources.rtmodels as rtmodels

from core.roadmap import ACCRoadMap
from core.roadmap import ACCDirector
from core.environment.exception import AnomalousEpisodeException
from demos.override_demo import prepare_map_info
from .car import PPCar, PPCarMP, PPCarCan

def generate_car(car: QLabsQCar, location: list[float], rotation: list[float]) -> None:
    QLabsRealTime().terminate_all_real_time_models()
    car.destroy()
    car.spawn_id(
        actorNumber=0,
        location=location,
        rotation=rotation,
        scale=[.1, .1, .1],
        configuration=0,
        waitForConfirmation=True
    )
    time.sleep(1)
    QLabsRealTime().start_real_time_model(rtmodels.QCAR_STUDIO)
    time.sleep(2) # wait for the state to change

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
            car.execute()
        except IndexError:
            print("Reached the goal!")
            car.halt_car()
            done = True
        except AnomalousEpisodeException as e:
            print(e)
            done = True

    qlabs.close()

# TODO: Refactor this function
def run_subject(
    shm_name_action: str, 
    shm_name_image: str,
    step_event, 
    activate_event, 
    stop_event, 
    episode_event
) -> None:
    qlabs: QuanserInteractiveLabs = QuanserInteractiveLabs()
    qlabs.open("localhost")
    roadmap: ACCRoadMap = ACCRoadMap()
    waypoint_sequence: np.ndarray = roadmap.generate_path([10, 4, 14, 20, 22, 10])

    while True:
        episode_event.wait()
        car: PPCarCan = PPCarCan(qlabs=qlabs, waypoint_sequence=waypoint_sequence)
        car.setup() # get the initial observation
        activate_event.set()

        while not stop_event.is_set():
            try:
                car.execute(shm_name_image, shm_name_action, step_event)
            except IndexError:
                car.halt_car()
                stop_event.set()
            except AnomalousEpisodeException as e:
                print(e)

        activate_event.clear()
        episode_event.clear()

    qlabs.close()

# TODO: Refactor this function
def run_observer(
        shm_name_action: str, 
        shm_name_image: str,
        step_event, 
        activate_event, 
        stop_event, 
        episode_event
    ) -> None:
    qlabs: QuanserInteractiveLabs = QuanserInteractiveLabs()
    qlabs.open("localhost")
    director: ACCDirector = ACCDirector(qlabs)
    shm_action: SharedMemory = SharedMemory(name=shm_name_action)
    shm_image: SharedMemory = SharedMemory(name=shm_name_image)
    data_shared: np.ndarray = np.ndarray((2,), dtype=np.float64, buffer=shm_action.buf)
    image_shared: np.ndarray = np.ndarray((410, 820, 3), dtype=np.uint8, buffer=shm_image.buf)

    print("Preparing map...")
    init_pos, _ = prepare_map_info(node_id=24)
    actors: dict = director.build_map(init_pos)
    file_path = os.path.join(os.getcwd(), "test_data")
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    max_episode: int = 1000
    max_step: int = 800    
    counter: int = 0
    for i in range(max_episode):
        episode_step: int = 0
        image_queue: Queue = Queue()
        generate_car(actors["car"][0], actors["car"][1], actors["car"][2])
        episode_event.set()
        print("\nGenerating QCar...")
        activate_event.wait()
        while not stop_event.is_set():
            if episode_step >= max_step:
                stop_event.set()
                break

            if step_event.is_set():
                data = data_shared.copy()
                image = image_shared.copy()
                image_queue.put(image)
                # cv2.imshow("Pure Pursuite View", image)
                # cv2.waitKey(1)
                # sys.stdout.write(f'\rAction: {data}{" " * 20}')
                # sys.stdout.flush()
                step_event.clear()
                episode_step += 1
        print(f"\nEpisode {i+1} complete with {episode_step} steps")

        print("Saving images...")
        while not image_queue.empty(): 
            image = image_queue.get()
            if cv2.countNonZero(image) == 0:
                continue 
            file_name = f"image_{counter}.jpg"
            cv2.imwrite(os.path.join(file_path, file_name), image)
            counter += 1
        np.copyto(data_shared, np.zeros(2))
        stop_event.clear()

    qlabs.close()
    shm_action.close()
    shm_action.unlink()
    shm_image.close()
    shm_action.unlink()

def start_test_mp() -> None:
    processes: List[Process] = []
    step_event = Event()
    activate_event = Event()
    stop_event= Event()
    episode_event = Event()

    shm_name_action: str = "action"
    shm_name_image: str = "image"
    shm_action: SharedMemory = SharedMemory(name=shm_name_action, create=True, size=16)
    shm_image: SharedMemory = SharedMemory(name=shm_name_image, create=True, size=410*820*3) 
    p1: Process = Process(
        target=run_subject,
        args=(
            shm_name_action, 
            shm_name_image, 
            step_event, 
            activate_event, 
            stop_event, 
            episode_event
        )
    )
    p2: Process = Process(
        target=run_observer,
        args=(
            shm_name_action, 
            shm_name_image, 
            step_event, 
            activate_event, 
            stop_event, 
            episode_event
        )
    )

    processes.append(p1)
    processes.append(p2)
    for p in processes:
        p.start()

    while True:
        time.sleep(100)