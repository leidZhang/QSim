import numpy as np

from typing import Tuple
from core.base_policy import BasePolicy
from core.control.keyboard import KeyboardController


class KeyboardPolicy(BasePolicy): 
    def __init__(self) -> None:
        self.controller: KeyboardController = KeyboardController()

    def execute(self) -> Tuple[np.ndarray, dict]:
        self.controller.execute()
        x_signal = self.controller.state['x_signal']
        y_signal = self.controller.state['y_signal']
        return np.array([x_signal, y_signal]), {}
