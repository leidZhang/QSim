# 3rd party imports
import keyboard
import numpy as np
# custom imports
from .constants import *


class KeyboardController:
    def __init__(self, slow_to_zero: bool = False) -> None:
        self.slow_to_zero: bool = slow_to_zero
        self.axises = {
            'x_signal': (['up', 'down'], MAX_X_AXIS_VALUE, X_AXIS_DECREASE),
            'y_signal': (['left', 'right'], MAX_Y_AXIS_VALUE, Y_AXIS_DECREASE),
        } # key pairs, press left to increase and press right to decrease
        self.state = {
            'x_signal': 0, 
            'y_signal': 0,
        }

    def normalize_signal(self, signal: int, max_signal: int) -> float:
        return signal / max_signal

    def to_zero(self, val) -> int:
        if val > 0:
            return val - NORMAL_DECREASE
        elif val < 0:
            return val + NORMAL_DECREASE
        else:
            return 0

    def handle_axis(self, key_pair: list, signal: int, rate: int, max_signal: int) -> None:
        if keyboard.is_pressed(key_pair[0]):
            signal += rate
            if signal > max_signal:
                signal = max_signal
        elif keyboard.is_pressed(key_pair[1]):
            signal -= rate
            if signal < -max_signal:
                signal = -max_signal
        else:
            signal = self.to_zero(signal) if self.slow_to_zero else 0

        return signal

    def execute(self) -> None:
        # handle signals for triggered key pairs
        for key, val in self.axises.items():
            key_pair, max_signal, rate = val[0], val[1], val[2]
            signal: int = self.state[key] * max_signal
            signal = self.handle_axis(key_pair, signal, rate, max_signal)
            self.state[key] = self.normalize_signal(signal, max_signal)
    
    def clear_input_buffer(self) -> None:
        while keyboard.read_event(suppress=True):
            pass
