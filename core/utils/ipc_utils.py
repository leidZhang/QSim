import time
from typing import Any, Dict
from queue import Empty, Full
from multiprocessing import Queue
# from multiprocessing.shared_memory import SharedMemory

import numpy as np


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

def fetch_latest_in_queue(data_queue: Queue) -> None:
    latest_data: Any = None
    # if not data_queue.empty():
    #     latest_data = data_queue.get()
    try:
        latest_data = data_queue.get_nowait()
    except Empty:
        pass
    return latest_data

def put_latest_in_queue(data: Any, data_queue: Queue) -> None:
    # print(f"put latest data {data}")
    try:
        data_queue.put_nowait(data)
    except Full:
        data_queue.get()
        data_queue.put(data)

