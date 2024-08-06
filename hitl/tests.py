from queue import Queue

from core.remote import UDPClient
from system.settings import IP, HITL_PORT


def run_hitl_client() -> None:
    try:
        state_queue: Queue = Queue(1)
        client: UDPClient = UDPClient(IP, HITL_PORT)
        while True:
            client.execute(state_queue)
            print(f"Client state: {state_queue.get()}")
    except Exception as e:
        print(f"Client error: {e}")

