from core.templates import BasePolicy
from core.control.keyboard import KeyboardController


class KeyboardPolicy(BasePolicy): 
    def __init__(self) -> None:
        self.controller: KeyboardController = KeyboardController()
        
    def execute(self) -> tuple:
        self.controller.read()
        v_slider: int = self.controller.state["v_slider"]
        x_signal: int = self.controller.state["x_signal"]
        return x_signal, v_slider