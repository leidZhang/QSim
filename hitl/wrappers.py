import time
import pickle
from typing import Any

from core.remote import UDPServer
from core.templates import LifeCycleWrapper
from system.settings import HITL_PORT, IP
from .modules import KeyboardPolicy


class KbdWrapper(LifeCycleWrapper):
    def __init__(self) -> None:
        self.server: UDPServer = UDPServer(IP, HITL_PORT)
        self.policy: KeyboardPolicy = KeyboardPolicy()
        self.action: int = 0

    def _handle_send_data(self, data: Any) -> None:
        serialized_data: bytes = pickle.dumps(data)  
        self.server.server_socket.sendto(serialized_data, self.server.address)

    def execute(self) -> None:
        start = time.time()
        self.action: int = self.policy.execute()
        self.action = 1 if self.action > 0 else 0
        print(self.action)
        self._handle_send_data(self.action)
        end = time.time() - start
        # time.sleep(max(0.0, 0.03 - end))
        time.sleep(max(0.0, 0.2 - end))

    def terminate(self) -> None:
        self.server.server_socket.close()
