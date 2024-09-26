from typing import Tuple, Dict
# 3rd party imports
import keyboard
# custom imports
from .constants import *


class KeyboardController:
    """
    The KeyboardController class is a class that reads the keyboard input and converts it to signals.

    Attributes:
    - slow_to_zero: bool, default=False
    - table_driven: list, default=[None, self.handle_slider, self.handle_axis]
    - buttons: Dict[str, Dict[str, tuple]], default={'slider': {'v_slider': ('space', MAX_V_SLIDER_VALUE, V_SLIDER_DECREASE)}, 
        'axis': {'x_signal': (['up', 'down'], MAX_X_AXIS_VALUE, X_AXIS_DECREASE), 'y_signal': (['left', 'right'], 
        MAX_Y_AXIS_VALUE, Y_AXIS_DECREASE)}, 'press': {}}
    - state: Dict[str, int], default={'x_signal': 0, 'y_signal': 0, 'v_slider': 0}
    """

    def __init__(self, slow_to_zero: bool = False) -> None:
        """
        The KeyboardController class constructor, initialize the controller with the given parameters.

        Parameters:
        - slow_to_zero: bool, default=False
        """
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
        """
        Add a key pair to control an axis.

        Parameters:
        - key: str: The key to control the axis
        - values: Tuple[list, int, int]: The key pair, max signal, and rate to control the axis

        Raises:
        - ValueError: Key pair must be a tuple of two keys
        - ValueError: Max signal must be a positive integer
        - ValueError: Rate must be a positive integer

        Returns:
        - None
        """
        if values[0] is None or len(values[0]) != 2:
            raise ValueError("Key pair must be a tuple of two keys")
        if values[1] is None or values[1] <= 0:
            raise ValueError("Max signal must be a positive integer")
        if values[2] is None or values[2] <= 0:
            raise ValueError("Rate must be a positive integer")
        
        self.buttons['axis'][key] = values

    def add_slider(self, key: str, values: Tuple[str, int, int]) -> None:
        """
        Add a key to control a slider.

        Parameters:
        - key: str: The key to control the slider
        - values: Tuple[str, int, int]: The key, max signal, and rate to control the slider

        Raises:
        - ValueError: Max signal must be a positive integer
        - ValueError: Rate must be a positive integer

        Returns:
        - None
        """
        if values[1] is None or values[1] <= 0:
            raise ValueError("Max signal must be a positive integer")
        if values[2] is None or values[2] <= 0:
            raise ValueError("Rate must be a positive integer")

        self.buttons['slider'][key] = values

    def _normalize_signal(self, signal: int, max_signal: int) -> float:
        """
        Normalize the signal to a value between -max_signal and max_signal.

        Parameters:
        - signal: int: The signal to be normalized

        Returns:
        - float: The normalized signal
        """
        return signal / max_signal

    def _to_zero(self, val: int, decrease_value: int = NORMAL_DECREASE) -> int:
        """
        To zero the value with a decrease value.

        Parameters:
        - val: int: The value to be decreased
        - decrease_value: int, default=NORMAL_DECREASE: The decrease value

        Returns:
        - int: The decreased value
        """
        if val > 0:
            return val - decrease_value
        elif val < 0:
            return val + decrease_value
        else:
            return 0

    def handle_axis(self, key_pair: list, axis_signal: int, rate: int, max_signal: int) -> int:
        """
        Handle the axis signal based on the key pair.

        Parameters:
        - key_pair: list: The key pair to control the axis
        - axis_signal: int: The current axis signal
        - rate: int: The rate to increase or decrease the axis signal
        - max_signal: int: The maximum signal value

        Returns:
        - int: The updated axis signal
        """
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
        """
        Handle the slider signal based on the key.

        Parameters:
        - key: str: The key to control the slider
        - slider_signal: int: The current slider signal value
        - rate: int: The rate to increase or decrease the slider signal
        - max_signal: int: The maximum signal value

        Returns:
        - int: The updated slider signal
        """
        if keyboard.is_pressed(key):
            slider_signal += rate
            if slider_signal > max_signal:
                slider_signal = max_signal
        else:
            slider_signal = 0

        return slider_signal

    def read(self) -> None:
        """
        Read the keyboard input and convert it to signals.

        Returns:
        - None
        """
        # Handle signals for key pairs control axis
        for _, type in self.buttons.items(): 
            for key, val in type.items():
                # key_pair, max_signal, rate = val[0], val[1], val[2]
                method_index: int = (1 if len(val) == 3 else 0) + (1 if len(val[0]) == 2 else 0)
                signal: int = self.state[key] * val[1]
                signal = self.table_driven[method_index](val[0], signal, val[2], val[1])
                self.state[key] = self._normalize_signal(signal, val[1])

