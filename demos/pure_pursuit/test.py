import os
from queue import Queue
from typing import List
from multiprocessing import Queue as MPQueue
from multiprocessing import Process

import numpy as np

from qvl.qlabs import QuanserInteractiveLabs
from core.roadmap.dispatcher import TaskDispacher
from demos.performance_environment import prepare_test_environment, destroy_map
from .modules import PurePursuiteCar, MockOptitrackClient
from .executions import TaskDispatcherExec, BaseProcessExec, RecordDataWriterExec
from .executions import PurePursuiteCarExec, BaseThreadExec
from .settings import START_NODE


def test_dispatch_task_to_car() -> None:
    destroy_map() # destroy all spawned actors
    prepare_test_environment(node_id=START_NODE)

    # initialize the IPC queues
    obs_queue: MPQueue = MPQueue()
    task_queue: MPQueue = MPQueue(5)
    # create a processes
    dispatcher_exec: BaseProcessExec = TaskDispatcherExec()
    data_writer_exec: BaseProcessExec = RecordDataWriterExec()
    dispatcher_process: Process = Process(target=dispatcher_exec.run_process, args=(task_queue,))
    data_writer_process: Process = Process(target=data_writer_exec.run_process, args=(obs_queue,))
    # start the processes
    dispatcher_process.start()
    data_writer_process.start()

    # create a car thread
    try:
        car_exec: BaseThreadExec = PurePursuiteCarExec(
            task_queue=task_queue,
            obs_queue=obs_queue
        )
        car_exec.run_thread()
    except Exception as e:
        print(e)
    finally:
        car_exec.car.halt_car()
        os._exit(0) # exit the process


def test_purepursuite_car() -> None:
    destroy_map() # destroy all spawned actors

    qlabs: QuanserInteractiveLabs = QuanserInteractiveLabs()
    qlabs.open('localhost') # connect to the UE4
    waypoints: np.ndarray = prepare_test_environment(node_id=4)
    car: PurePursuiteCar = PurePursuiteCar(qlabs=qlabs, throttle_coeff=0.08)
    car.setup(waypoints=waypoints, init_waypoint_index=0)

    try:
        while True:
            car.execute()
    except Exception as e:
        print(e)
    finally:
        car.halt_car()


def test_mock_optitrack_client() -> None:
    destroy_map() # destroy all spawned actors

    qlabs: QuanserInteractiveLabs = QuanserInteractiveLabs()
    qlabs.open('localhost') # connect to the UE4
    prepare_test_environment(node_id=START_NODE)

    data_queue: Queue = Queue(5)
    client: MockOptitrackClient = MockOptitrackClient(data_queue=data_queue)
    try:
        while True:
            client.execute(qlabs)
            print(data_queue.get())
    except Exception as e:
        print(e)


def test_dispatcher_exec() -> None:
    queue: MPQueue = MPQueue(5)
    exec: BaseProcessExec = TaskDispatcherExec()
    process: Process = Process(target=exec.run_process, args=(queue,))
    process.start()

    counter: int = 0
    while counter <= 100:
        if not queue.empty():
            data: List[int] = queue.get()
            print(f"Get new task: {data[0]}")
            counter += 1

    print("End of test_dispatcher_exec()")
    os._exit(0) # exit the process
