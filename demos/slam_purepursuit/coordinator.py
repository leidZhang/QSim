import os
import time
from queue import Queue
from threading import Thread
from multiprocessing import Process
from multiprocessing import Queue as MPQueue
from typing import List, Dict, Union, Any

import numpy as np

from core.utils.executions import BaseCoordinator
from core.utils.executions import BaseThreadExec, BaseProcessExec
from demos.traditional.executions import ObserveExec
from .executions import PurepursuitCarComm, ObservationComm
from .executions import WaypointProcessorExec, ResNetDetectorExec


class LidarSLAMCoordinator(BaseCoordinator):
    def __init__(self, queue_size: int = 5) -> None:
        super().__init__() # create self.pool
        self.queues: Dict[str, Union[Queue, MPQueue]] = {
            'waypoint_request': MPQueue(queue_size),
            'waypoint_response': MPQueue(queue_size), # 'observe_response' -> 'observation_response
            'detection_request': MPQueue(queue_size),
            'detection_response': MPQueue(queue_size),
            'obastacle_request': MPQueue(queue_size),
            'obstacle_response': MPQueue(queue_size),
        } # queues for IPC
        self.settings: Dict[str, dict] = {
            'thread': {
                'obs_comm': (ObservationComm(debug=True), (self.queues['detection_response'], self.queues['waypoint_response'], self.queues['obstacle_response'])),
                'car_comm': (PurepursuitCarComm(), (self.queues['detection_request'], self.queues['waypoint_request'], self.queues['obastacle_request']))
            }, # required args for thread
            'process': {
                'observe': (ObserveExec(), (self.queues['detection_response'], self.queues['detection_request'])),
                'waypoint': (WaypointProcessorExec(), (self.queues['waypoint_response'], self.queues['waypoint_request'])),
                'detector': (ResNetDetectorExec(), (self.queues['obstacle_response'], self.queues['obastacle_request'])),
            } # required args for process
        }

    def terminate(self) -> None:
        # set the main events
        for types in self.settings.keys():
            for val in self.settings[types].values():
                # extract exec instance from setting
                exec: Union[BaseThreadExec, BaseProcessExec] = val[0]
                # terminate the threads or proess by calling terminate
                exec.terminate()

        # join the threads and process
        for types in self.pools.keys():
            for instance in self.pools[types]:
                instance.join()

    def start_main_process(self) -> None:
        print("Using parallel method")
        # add the sub-processes and threads to the pool
        for exec_type in self.settings.keys():
            for exec_name in self.settings[exec_type]:
                self._add_to_pool(exec_type=exec_type, exec_name=exec_name)

        # activate the sub-processes
        for process in self.pools['process']:
            process.start()
        time.sleep(6)
        # activate the threads
        for threads in self.pools['thread']:
            threads.start()

        # observe keyboard interrupt
        # self.observe_keyboard_interrupt()
        self.run_monitor_thread()

    start = start_main_process # alias

    def observe_keyboard_interrupt(self) -> None:
        while True:
            try:
                time.sleep(100)
            except KeyboardInterrupt:
                self.terminate()
                os._exit(0)

    def _add_to_pool(self, exec_type: str, exec_name: str) -> None:
        # extract parameters from the setting dict
        setting: dict = self.settings[exec_type]
        exec: Union[BaseThreadExec, BaseProcessExec] = setting[exec_name][0]
        args: tuple = setting[exec_name][1]

        # create process or thread
        result: Union[Thread, Process] = None
        if exec_type == 'thread':
            # create the thread
            result = Thread(target=exec.run_thread, name=exec_name, args=args)
        elif exec_type == 'process':
            # create the process
            result = Process(target=exec.run_process, name=exec_name, args=args)
        else:
            raise ValueError("Invaild exec type, exec type is either 'thread' or 'process'")

        # add thread or process to the pool
        self.pools[exec_type].append(result)