from typing import Any, Dict
from multiprocessing.shared_memory import SharedMemory

import numpy as np


class StructedDataTypeFactory:
    def create_dtype(self, num_of_cmds: int, image_size: tuple) -> np.ndarray:
        data_type: np.dtype = np.dtype([
            ('timestamp', np.float64),
            ('data_and_commands', np.float64, (num_of_cmds, )),
            ('image', np.uint8, image_size),
        ])
        data = np.zeros(1, dtype=data_type)
        return data


class SharedMemoryWrapper:
    def __init__(self, data: Any, shm_name: str, create: bool) -> None:
        if create: # create new shared memory
            self.shared_memory: SharedMemory = SharedMemory(name=shm_name, create=True, size=data.nbytes)
        else: # connect to the existing shared memory
            self.shared_memory: SharedMemory = SharedMemory(name=shm_name, create=False, size=data.nbytes)
        self.shared_data: np.ndarray = np.ndarray(data.shape, dtype=data.dtype, buffer=self.shared_memory.buf)

    def write_to_shm(self, key: str, new_value: Any) -> None:
        self.shared_data[0][key] = new_value

    def read_from_shm(self, key: str) -> Any:
        return self.shared_data[0][key]