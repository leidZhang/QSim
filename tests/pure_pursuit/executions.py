import time
from threading import Event
from typing import Tuple, List
from multiprocessing import Queue

import numpy as np

from qvl.qlabs import QuanserInteractiveLabs
from core.utils.executions import BaseProcessExec
from core.utils.executions import BaseThreadExec
from core.roadmap.dispatcher import TaskDispacher
from .modules import PurePursuiteCar, RecordDataWriter
from .settings import START_NODE


class RecordDataWriterExec(BaseProcessExec):
    def create_instance(self) -> RecordDataWriter:
        return RecordDataWriter(folder_path='test_data')


class TaskDispatcherExec(BaseProcessExec):
    def create_instance(self) -> TaskDispacher:
        return TaskDispacher(start_node=START_NODE)


class PurePursuiteCarExec(BaseThreadExec):
    def __init__(
        self, 
        task_queue: Queue, 
        obs_queue: Queue, 
        watchdog_event: Event = None
    ) -> None:
        super().__init__(watchdog_event)
        self.obs_queue: Queue = obs_queue
        self.task_queue: Queue = task_queue
        self.car: PurePursuiteCar = None

    def setup_thread(self) -> None:
        # wait for the task data from the dispatcher
        while self.task_queue.empty():
            time.sleep(0.1)
        # decode the task data
        task_data: Tuple[List[int], np.ndarray] = self.task_queue.get()
        node_sequence: List[int] = task_data[0] # get the node sequence data
        waypoints: np.ndarray = task_data[1] # get the waypoints data
        # connect to the qlab
        qlabs: QuanserInteractiveLabs = QuanserInteractiveLabs()
        qlabs.open('localhost') # connect to the UE4
        # create the car instance
        self.car = PurePursuiteCar(qlabs=qlabs, throttle_coeff=0.08)
        self.car.setup(
            node_sequence=node_sequence,
            waypoints=waypoints, 
            init_waypoint_index=0
        )

    def execute(self) -> None:
        self.car.execute(self.task_queue, self.obs_queue)

    def run_thread(self) -> None:
        super().run_thread()
        self.car.halt_car()