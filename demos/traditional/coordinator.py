import time
from threading import Thread
from threading import Event as ThEvent
from multiprocessing import Event as MpEvent
from multiprocessing import Queue, Process
from typing import List, Dict, Union, Any

from core.utils.executions import Event
from core.utils.executions import BaseCoordinator, WatchDogTimer
from core.utils.executions import BaseThreadExec, BaseProcessExec
from .executions import EdgeFinderExec, ObserveExec
from .executions import EdgeFinderComm, ObserveComm, CarComm


class QCarCoordinator(BaseCoordinator):
    def __init__(self, queue_size: int = 5) -> None:
        # events for monitoring
        events: Dict[str, Event] = {
            'edge_finder_comm': ThEvent(),
            'observe_comm': ThEvent(),
            'car_comm': ThEvent(),
            'edge_finder': MpEvent(),
            'observe': MpEvent()
        }
        # watchdogs to monitor the threads and processes
        watchdogs: Dict[str, Dict[str, WatchDogTimer]] = {
            'thread': {
                'edge_finder_comm': WatchDogTimer(event=events['edge_finder_comm'], timeout=0.05),
                # 'observe_comm': WatchDogTimer(event=events['observe_comm'], timeout=0.05),
                'car_comm': WatchDogTimer(event=events['car_comm'], timeout=4), # temp timeout
            },
            'process': {
                'edge_finder': WatchDogTimer(event=events['edge_finder'], timeout=0.3),
                'observe': WatchDogTimer(event=events['observe'], timeout=0.3)
            }
        }

        # create pool, watchdogs
        super().__init__(watchdogs=watchdogs)

        # queues for IPC
        self.queues: Dict[str, Queue] = {
            'edge_request': Queue(queue_size),
            'edge_response': Queue(queue_size),
            'observe_request': Queue(queue_size),
            'observe_response': Queue(queue_size)
        }
        # required args for subprocesses and threads
        self.settings: Dict[str, dict] = {
            'thread': {
                'edge_finder_comm': (EdgeFinderComm(events['edge_finder_comm']), (self.queues['edge_response'], self.queues['observe_response'], )),
                # 'observe_comm': (ObserveComm(events['observe_comm']), (self.queues['observe_response'], )),
                'car_comm': (CarComm(events['car_comm']), (self.queues['edge_request'], self.queues['observe_request'], ))
            }, # required args for thread
            'process': {
                'edge_finder': (EdgeFinderExec(events['edge_finder']), (self.queues['edge_response'], self.queues['edge_request'], )),
                'observe': (ObserveExec(events['observe']), (self.queues['observe_response'], self.queues['observe_request'], ))
            } # required args for process
        }

    def terminate(self) -> None:
        # set the main events
        for types in self.settings.keys():
            for key, val in self.settings[types].items():
                # extract exec instance from setting
                exec: Union[BaseThreadExec, BaseProcessExec] = val[0]
                print(f"terminating {types} {key}...")
                # terminate the threads or proess by calling terminate
                exec.terminate()
                print(f"{types} {key} terminated")

        # join the threads and process
        for types in self.pools.keys():
            for instance in self.pools[types]:
                instance.join()

    def start_main_process(self) -> None:
        # add the sub-processes and threads to the pool
        for exec_type in self.settings.keys():
            for exec_name in self.settings[exec_type]:
                self._add_to_pool(exec_type=exec_type, exec_name=exec_name)

        # activate the sub-processes
        for process in self.pools['process']:
            process.start()
        # wait until process are activated
        self.prepare_processes()
        # activate the threads
        for threads in self.pools['thread']:
            threads.start()

        # observe the processes and threads
        self.run_monitor_thread()

    start = start_main_process # alias

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
