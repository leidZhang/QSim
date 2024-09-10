import numpy as np

from typing import Tuple
from core.templates.base_policy import BasePolicy
from core.control.keyboard import KeyboardController


class KeyboardPolicy(BasePolicy): 
    def __init__(self) -> None:
        self.controller: KeyboardController = KeyboardController()

    def execute(self) -> Tuple[np.ndarray, dict]:
        self.controller.read()
        x_signal = self.controller.state['x_signal'] # up dand down
        y_signal = self.controller.state['y_signal'] # left and right
        v_slider = self.controller.state['v_slider'] # space bar
        return np.array([x_signal * (1 - v_slider), y_signal]), {}
