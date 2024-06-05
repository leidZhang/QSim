from keyboard import KeyboardEvent
from abc import ABC, abstractmethod


class KeyboardControllerStrategy(ABC):
    """
    Abstract base class for keyboard control strategies for a QCar model.
    """

    def __init__(self, state: dict, key: str) -> None:
        self.state: dict = state
        self.key: str = key

    @abstractmethod
    def execute(self, keyboard_event: KeyboardEvent) -> None:
        pass


class KeyboardReverseFlagStrategy(KeyboardControllerStrategy):
    """
    Strategy to toggle the reverse flag in the QCar's control state when a specific key is pressed.

    Overrides:
        execute(keyboard_event: KeyboardEvent): Toggles the 'reverse' flag in the QCar's control state.
    """

    def execute(self, keyboard_event: KeyboardEvent) -> None:
        if keyboard_event.event_type == 'down':
            self.state['control_flags']['reverse'] = not self.state['control_flags']['reverse']


class KeyboardCruiseFlagStrategy(KeyboardControllerStrategy):
    """
    Strategy to toggle the cruise control flag and set the cruise throttle in the QCar's control state when a specific key is pressed.

    Overrides:
        execute(keyboard_event: KeyboardEvent): Toggles the 'cruise' flag and sets the 'cruise_throttle' in the QCar's control state.
    """

    def execute(self, keyboard_event: KeyboardEvent) -> None:
        if keyboard_event.event_type == 'down':
            self.state['control_flags']['cruise'] = not self.state['control_flags']['cruise']
            self.state['cruise_throttle'] = self.state['throttle']


class KeyboardLightFlagStrategy(KeyboardControllerStrategy):
    """
    Strategy to toggle the light flag in the QCar's control state when a specific key is pressed.

    Overrides:
        execute(keyboard_event: KeyboardEvent): Toggles the 'light' flag in the QCar's control state.
    """

    def execute(self, keyboard_event: KeyboardEvent) -> None:
        if keyboard_event.event_type == 'down':
            self.state['control_flags']['light'] = not self.state['control_flags']['light']


class KeyboardSafeFlagStrategy(KeyboardControllerStrategy):
    """
    Strategy to toggle the safety flag in the QCar's control state when a specific key is pressed.

    Overrides:
        execute(keyboard_event: KeyboardEvent): Toggles the 'safe' flag in the QCar's control state.
    """

    def execute(self, keyboard_event: KeyboardEvent) -> None:
        if keyboard_event.event_type == 'down':
            self.state['control_flags']['safe'] = not self.state['control_flags']['safe']
