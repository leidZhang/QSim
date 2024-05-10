import os
from queue import Queue
from typing import Tuple
from collections import deque

from core.policies.keyboard import KeyboardPolicy


class OverrideDetector: 
    def __init__(self) -> None:
        self.signal_buffer: deque = deque(maxlen=10)
        self.signal_accumulator: float = 0.0
        self.diff_buffer_1: Queue = Queue(5)
        self.diff_buffer_2: Queue = Queue(5)

    def swap_diff_buffer(self, buffer_1: Queue, buffer_2: Queue) -> None:
        self.diff_buffer_1 = buffer_2
        self.diff_buffer_2 = buffer_1

    def get_signal(self, signal: float) -> None: 
        if len(self.signal_buffer) == 10: 
            self.signal_accumulator -= self.signal_buffer.pop()
        self.signal_buffer.appendleft(signal)
        self.signal_accumulator += signal

def run_keyboard():
    policy: KeyboardPolicy = KeyboardPolicy(slow_to_zero=True)
    while True:
        state = policy.execute()
        print(state)
        os.system("cls")