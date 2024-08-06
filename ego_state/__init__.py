import time
from threading import Thread
from .executions import RelayExecutor

print("Initializing relay...")
relay: RelayExecutor = RelayExecutor()
relay_thread: Thread = Thread(target=relay.run, daemon=True)