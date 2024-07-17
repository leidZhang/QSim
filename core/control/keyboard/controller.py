import time
from typing import Tuple, Dict
# 3rd party imports
import keyboard
# custom imports
from .constants import *


class KeyboardController:
    def __init__(self, slow_to_zero: bool = False) -> None:
        self.slow_to_zero: bool = slow_to_zero
        self.table_driven: list = [
            None, # placeholder for button press
            self.handle_slider,
            self.handle_axis,
        ]
        self.buttons: Dict[str, Dict[str, tuple]] = {
            'slider': {
                'v_slider': ('space', MAX_V_SLIDER_VALUE, V_SLIDER_DECREASE),
            }, # key, press to increase
            'axis': {
                'x_signal': (['up', 'down'], MAX_X_AXIS_VALUE, X_AXIS_DECREASE),
                'y_signal': (['left', 'right'], MAX_Y_AXIS_VALUE, Y_AXIS_DECREASE),
            }, # key pairs, press left to increase and press right to decrease
            'press': {},
        }
        self.state = {
            'x_signal': 0, 
            'y_signal': 0,
            'v_slider': 0,
        }

    def add_axis(self, key: str, values: Tuple[list, int, int]) -> None:
        if values[0] is None or len(values[0]) != 2:
            raise ValueError("Key pair must be a tuple of two keys")
        if values[1] is None or values[1] <= 0:
            raise ValueError("Max signal must be a positive integer")
        if values[2] is None or values[2] <= 0:
            raise ValueError("Rate must be a positive integer")
        
        self.buttons['axis'][key] = values

    def add_slider(self, key: str, values: Tuple[str, int, int]) -> None:
        if values[1] is None or values[1] <= 0:
            raise ValueError("Max signal must be a positive integer")
        if values[2] is None or values[2] <= 0:
            raise ValueError("Rate must be a positive integer")

        self.buttons['slider'][key] = values

    def _normalize_signal(self, signal: int, max_signal: int) -> float:
        return signal / max_signal

    def _to_zero(self, val: int, decrease_value: int = NORMAL_DECREASE) -> int:
        if val > 0:
            return val - decrease_value
        elif val < 0:
            return val + decrease_value
        else:
            return 0

    def handle_axis(self, key_pair: list, axis_signal: int, rate: int, max_signal: int) -> int:
        if keyboard.is_pressed(key_pair[0]):
            axis_signal += rate
            if axis_signal > max_signal:
                axis_signal = max_signal
        elif keyboard.is_pressed(key_pair[1]):
            axis_signal -= rate
            if axis_signal < -max_signal:
                axis_signal = -max_signal
        else:
            axis_signal = self._to_zero(axis_signal, rate) if self.slow_to_zero else 0

        return axis_signal
    
    def handle_slider(self, key: str, slider_signal: int, rate: int, max_signal: int) -> int:
        if keyboard.is_pressed(key):
            slider_signal += rate
            if slider_signal > max_signal:
                slider_signal = max_signal
        else:
            slider_signal = 0

        return slider_signal

    def read(self) -> None:
        # Handle signals for key pairs control axis
        for _, type in self.buttons.items(): 
            for key, val in type.items():
                # key_pair, max_signal, rate = val[0], val[1], val[2]
                method_index: int = (1 if len(val) == 3 else 0) + (1 if len(val[0]) == 2 else 0)
                signal: int = self.state[key] * val[1]
                signal = self.table_driven[method_index](val[0], signal, val[2], val[1])
                self.state[key] = self._normalize_signal(signal, val[1])

        # # Handle signals for sliders
        # for key, val in self.sliders.items():
        #     button, max_signal, rate = val[0], val[1], val[2]
        #     signal: int = self.state[key] * max_signal
        #     signal = self.handle_slider(button, signal, rate, max_signal)
        #     self.state[key] = self._normalize_signal(signal, max_signal)
