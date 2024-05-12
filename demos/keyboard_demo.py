import os
from queue import Queue
from typing import Tuple
from collections import deque, defaultdict

from core.policies.keyboard import KeyboardPolicy


class OverrideDetector:
    def __init__(self) -> None:
        # signal buffer to store the last 10 signals
        self.signal_buffer: deque = deque(maxlen=10)
        self.signal_buffer.append(0) # mock signal to initialize the buffer
        self.signal_accumulator: float = 0.0
        # signal difference buffer to store the last 5 signal differences
        self.diff_buffer_1: Queue = Queue(5)
        self.diff_buffer_2: Queue = Queue(5)
        self.diff_accumulator: float = 0.0

    def swap_diff_buffer(self, buffer_1: Queue, buffer_2: Queue) -> None:
        self.diff_buffer_1 = buffer_2
        self.diff_buffer_2 = buffer_1

    def handle_diff_buffer(self, diff: float) -> dict:
        diff_dict: dict = {}
        if self.diff_buffer_1.full():
            pop_diff = self.diff_buffer_1.get()
            self.diff_accumulator -= pop_diff
        self.diff_buffer_1.put(diff)
        self.diff_accumulator += diff

        while not self.diff_buffer_1.empty():
            diff: int = self.diff_buffer_1.get()
            diff_dict[diff] = diff_dict.get(diff, 0) + 1
            self.diff_buffer_2.put(diff)
        # swap the queue for double
        self.swap_diff_buffer(self.diff_buffer_1, self.diff_buffer_2)

        return diff_dict

    def get_signal(self, signal: float) -> dict:
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

    def is_dominate(self, diff: int, diff_dict: dict) -> bool:
        queue_size: int = max(self.diff_buffer_1.qsize(), self.diff_buffer_2.qsize())
        if diff in diff_dict.keys():
            return diff_dict[diff] / queue_size >= 0.8
        return False

    def __call__(self, signal: int) -> bool:
        is_override: bool = False
        diff_dict: dict = self.get_signal(signal)
        # check if the difference is significant
        if signal > 0:
            is_override: bool = self.is_dominate(-5.0, diff_dict)
        elif signal < 0:
            is_override: bool = self.is_dominate(+5.0, diff_dict)
        else:
            is_override: bool = False

        return is_override

def run_keyboard():
    detector: OverrideDetector = OverrideDetector()
    policy: KeyboardPolicy = KeyboardPolicy(slow_to_zero=True)
    while True:
        state = policy.execute()
        print(f"State: {state[1] * 500}")
        print(f"Override status: {detector(state[1] * 500)}")
        os.system("cls")
