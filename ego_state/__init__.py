from multiprocessing import Queue, Event
from .wrapper import RelayWrapper


def run_relay(state_queue: Queue, event) -> None:
    print("Initializing relay...")
    relay: RelayWrapper = RelayWrapper()
    while True:
        relay.execute(state_queue, event)
    # relay_thread: Thread = Thread(target=relay.run, daemon=True)