import os
from queue import Queue
from typing import Tuple
from collections import deque, defaultdict

from core.policies.keyboard import KeyboardPolicy


class OverrideDetector: 
    def __init__(self) -> None:
        # signal buffer to store the last 10 signals
        self.signal_buffer: deque = deque(maxlen=10)
        self.signal_buffer.append(0)
        self.signal_accumulator: float = 0.0
        # signal difference buffer to store the last 5 signal differences
        self.diff_buffer_1: Queue = Queue(5)
        self.diff_buffer_2: Queue = Queue(5)

    def swap_diff_buffer(self, buffer_1: Queue, buffer_2: Queue) -> None:
        self.diff_buffer_1 = buffer_2
        self.diff_buffer_2 = buffer_1

    def handle_diff_buffer(self, diff: int) -> dict: 
        if self.diff_buffer_1.full():
            self.diff_buffer_1.get()
        self.diff_buffer_1.put(diff)

        diff_dict: dict = {}
        while not self.diff_buffer_1.empty():
            diff: int = self.diff_buffer_1.get()
            diff_dict[diff] = diff_dict.get(diff, 0) + 1
            self.diff_buffer_2.put(diff)
        # swap the queue for double
        self.swap_diff_buffer(self.diff_buffer_1, self.diff_buffer_2)

        return diff_dict

    def get_signal(self, signal: int) -> dict: 
        # calculate the signal difference
        left_signal: int = self.signal_buffer[0]
        signal_diff: int = signal - left_signal
        # push the signal to the signal buffer
        if len(self.signal_buffer) == 10: 
            self.signal_accumulator -= self.signal_buffer.pop()
        self.signal_buffer.appendleft(signal)
        self.signal_accumulator += signal
        # push the signal difference to the diff buffer
        return self.handle_diff_buffer(signal_diff)

    def __call__(self, signal: int) -> bool:
        is_override: bool = False
        diff_dict: dict = self.get_signal(signal)
        # check if the difference is significant
        if -5 in diff_dict.keys() and 5 in diff_dict.keys():
            is_override: bool = diff_dict[-5] / len(diff_dict) >= 0.6
            is_override = is_override or diff_dict[5] / len(diff_dict) >= 0.6

        return is_override

def run_keyboard():
    detector: OverrideDetector = OverrideDetector()
    policy: KeyboardPolicy = KeyboardPolicy(slow_to_zero=True)
    while True:
        state = policy.execute()
        print(detector(state[0]))
        os.system("cls")