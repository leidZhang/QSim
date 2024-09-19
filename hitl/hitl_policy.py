from core.templates import BasePolicy
from core.control.keyboard import KeyboardController


class KeyboardPolicy(BasePolicy):
    def __init__(self) -> None:
        self.controller: KeyboardController = KeyboardController(slow_to_zero=True)

    def execute(self) -> tuple:
        self.controller.read()
        v_slider: int = self.controller.state["v_slider"]
        x_signal: int = self.controller.state["x_signal"]

        accelerate: float = 1.0 if x_signal > 0 else 0.0
        break_pedal: float = 1.0 if v_slider > 0 else 0.0
        return accelerate, break_pedal