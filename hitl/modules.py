from core.templates import BasePolicy
from core.control.keyboard import KeyboardController


class KeyboardPolicy(BasePolicy): 
    def __init__(self) -> None:
        self.controller: KeyboardController = KeyboardController()
        
    def execute(self) -> int:
        self.controller.read()
        return self.controller.state['v_slider']