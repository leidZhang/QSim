from threading import Thread
from multiprocessing import Queue
from typing import List, Dict, Tuple, Any

from core.qcar import VirtualCSICamera, VirtualRGBDCamera
from .factory import PIDControlCarFactory


# TODO: Implement this class
class QCarCoordinator:
    def __init__(self, queue_size: int = 5) -> None:
        # CREATE QUANSER HARDWARE OBJECTS IN A THREAD OR A PROCESS
        self.thread_pool: List[Thread] = []
        self.queues: Dict[str, Queue] = {
            'edge_request': Queue(queue_size),
            'edge_response': Queue(queue_size),
            'observe_request': Queue(queue_size),
            'observe_response': Queue(queue_size)
        }
        self.comm_settings: Dict[str, Tuple[Any, tuple]] = {
            'edge_finder_comm': (VirtualCSICamera, (self.queues['edge_response'], )),
            'observe_comm': (VirtualRGBDCamera, (self.queues['observe_response'], )),
            'car_comm': (PIDControlCarFactory(), (self.queues['edge_request'], self.queues['observe_request']))
        } # required variables for comm factory and thread

    def add_to_thread_pool(self, target, args: Any) -> None:
        thread: Thread = Thread(target=target, args=args)
        self.thread_pool.append(thread)

    def start_main_process(self) -> None:
        ...
        # threads.append(Thread(target=run_edge_finder_comm))
        # threads.append(Thread(target=run_observe_comm))
        # print("Starting the demo...")
        # for thread in threads:
        #     thread.start()
        # run_car_comm()