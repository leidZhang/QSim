from typing import Callable, Any

from .hitl_policy import KeyboardPolicy


def run_keyboard_policy(callback: Callable = lambda *args: None, *args: Any) -> None:
    while True:
        hitl: KeyboardPolicy = KeyboardPolicy()
        accelerate, break_pedal = hitl.execute()
        print(accelerate, break_pedal)
        callback(accelerate, break_pedal, *args)
