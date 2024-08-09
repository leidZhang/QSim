import time
from threading import Lock

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


class RelayExecutor:
    def __init__(self) -> None:
        self.wrapper: EgoStateRelay = EgoStateRelay()
        self.done: bool = False

    def run(self) -> None:
        while not self.done:
            start: float = time.time()
            self.wrapper.handle_read_ego_state()
            self.wrapper.handle_read_bot_state()
            end: float = time.time() - start
            time.sleep(max(0.0, DT - end))

    def get_ego_states(self) -> None:
        return self.wrapper.get_ego_states()
    
    def set_early_stop(self, early_stop: bool, lock: Lock) -> None:
        with lock:
            self.wrapper.early_stop = early_stop

    def get_early_stop(self) -> bool:
        return self.wrapper.early_stop
    
    def terminate(self) -> None:
        self.done = True
        self.wrapper.terminate()
        print("Ego state relay terminated")
