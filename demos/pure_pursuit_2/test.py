import os
from queue import Queue
from typing import List
from multiprocessing import Queue as MPQueue
from multiprocessing import Process

import numpy as np

from core.roadmap.dispatcher import TaskDispacher
from .modules import PurePursuiteCar
from .executions import TaskDispatcherExec, BaseProcessExec, RecordDataWriterExec
from .executions import PurePursuiteCarExec, BaseThreadExec
from .settings import START_NODE


def test_dispatch_task_to_car_2() -> None:
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
