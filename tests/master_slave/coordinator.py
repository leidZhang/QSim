from threading import Thread
from multiprocessing import Queue, Process
from typing import List, Dict, Union, Any

from core.utils.executions import BaseThreadExec, BaseProcessExec
from .executions import EdgeFinderExec, ObserveExec
from .executions import EdgeFinderComm, ObserveComm, CarComm


# TODO: Implement activation method
class QCarCoordinator:
    def __init__(self, queue_size: int = 5) -> None:
        # CREATE QUANSER HARDWARE OBJECTS IN A THREAD OR A PROCESS
        self.pools: Dict[str, list] = {
            'thread': [],
            'process': []
        }
        self.queues: Dict[str, Queue] = {
            'edge_request': Queue(queue_size),
            'edge_response': Queue(queue_size),
            'observe_request': Queue(queue_size),
            'observe_response': Queue(queue_size)
        } # queues for IPC
        self.settings: Dict[str, dict] = {
            'thread': {
                'edge_finder_comm': (EdgeFinderComm(), (self.queues['edge_response'], )),
                'observe_comm': (ObserveComm(), (self.queues['observe_response'], )),
                'car_comm': (CarComm(), (self.queues['edge_request'], self.queues['observe_request']))
            }, # required args for thread
            'process': {
                'edge_finder': (EdgeFinderExec(), (self.queues['edge_response'], self.queues['edge_request'])),
                'observe': (ObserveExec(), (self.queues['observe_response'], self.queues['observe_request']))
            } # required args for process
        }

    def add_to_pool(self, exec_type: str, exec_name: str) -> None:
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

    def terminate(self) -> None:
        # set the main events
        for types in self.settings.keys():
            for val in self.settings[types].values():
                # extract exec instance from setting
                exec: Union[BaseThreadExec, BaseProcessExec] = val[0]
                # set the event to terminate the threads or proess by calling terminate
                exec.terminate()

        # join the threads and process
        for types in self.pools.keys():
            for instance in self.pools[types]:
                instance.join()

    def start_main_process(self) -> None:
        ...
        # threads.append(Thread(target=run_edge_finder_comm))
        # threads.append(Thread(target=run_observe_comm))
        # print("Starting the demo...")
        # for thread in threads:
        #     thread.start()
        # run_car_comm()