import time
from typing import Any, Tuple
from queue import Empty, Full
from multiprocessing import Queue, Event
# from multiprocessing.shared_memory import SharedMemory


def fetch_latest_in_queue(data_queue: Queue) -> None:
    latest_data: Any = None
    try:
        latest_data = data_queue.get_nowait()
    except Empty:
        pass
    return latest_data


def put_latest_in_queue(data: Any, data_queue: Queue) -> None:
    try:
        data_queue.put_nowait(data)
    except Full:
        data_queue.get()
        data_queue.put(data)


def clear_queue(data_queue: Queue) -> None:
    while not data_queue.empty():
        data_queue.get()


class EventQueue:
    def __init__(self, size: int = 1) -> None:
        self.buffer: Queue = Queue(size)
        self.queue: Queue = Queue(size)
        self.event = Event() # multiprocessing event

    def _switch_queue_and_buffer(self) -> Tuple[Queue, Queue]:
        temp: Queue = self.queue
        self.queue = self.buffer
        self.buffer = temp

    def put(self, data: Any) -> None:
        if self.buffer.full():
            self.buffer.get()
        self.buffer.put(data)

        if not self.event.is_set():
            self.event.set()

    def get(self) -> Any:
        if self.event.is_set():
            self.event.clear()
            self._switch_queue_and_buffer()
            return self.queue.get()
        return None

# class StructedDataTypeFactory:
#     def create_dtype(self, num_of_cmds: int, image_size: tuple) -> np.ndarray:
#         data_type: np.dtype = np.dtype([
#             ('timestamp', np.float64),
#             ('data_and_commands', np.float64, (num_of_cmds, )),
#             ('image', np.uint8, image_size),
#         ])
#         data = np.zeros(1, dtype=data_type)
#         return data


# class SharedMemoryWrapper:
#     def __init__(self, data: Any, shm_name: str, create: bool) -> None:
#         if create: # create new shared memory
#             self.shared_memory: SharedMemory = SharedMemory(name=shm_name, create=True, size=data.nbytes)
#         else: # connect to the existing shared memory
#             self.shared_memory: SharedMemory = SharedMemory(name=shm_name, create=False, size=data.nbytes)
#         self.shared_data: np.ndarray = np.ndarray(data.shape, dtype=data.dtype, buffer=self.shared_memory.buf)

#     def write_to_shm(self, key: str, new_value: Any) -> None:
#         self.shared_data[0][key] = new_value

#     def read_from_shm(self, key: str) -> Any:
#         return self.shared_data[0][key]