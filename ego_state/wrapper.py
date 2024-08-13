import time
from queue import Queue
from typing import List
from threading import Event

import numpy as np

from system.settings import DT
from .modules import EgoStateRelay


# def run_ego_state_relay(done) -> None:
#     try:
#         relay: EgoStateRelay = EgoStateRelay()
#         while not done.is_set():
#             relay.execute()
#         relay.terminate()
#     except KeyboardInterrupt:
#         relay.terminate()
#         print("Ego state relay terminated")


class RelayWrapper:
    def __init__(self) -> None:
        self.wrapper: EgoStateRelay = EgoStateRelay()
        self.flush_flag: bool = False

    def execute(self, state_queue: Queue, event: Event) -> None:
        # while not self.done:
        start: float = time.time()
        if not event.is_set():
            self.wrapper.handle_read_ego_state()
        else:
            self.wrapper.handle_reset_signal()
        self.wrapper.handle_read_bot_state()
        ego_states: List[np.ndarray] = self.get_ego_states()  
        if event.is_set():
            print(f"Ego state: {ego_states[0]}")  
        if state_queue.full():
            state_queue.get()
        state_queue.put(ego_states)
        end: float = time.time() - start
        event.wait(max(0.0, DT - end))

    def get_ego_states(self) -> None:
        return self.wrapper.get_ego_states()
    
    def terminate(self) -> None:
        self.done = True
        self.wrapper.terminate()
        print("Ego state relay terminated")
