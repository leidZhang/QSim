from typing import Any, Dict
from core.control.automata import BaseState
from core.utils.ipc_utils import DoubleBuffer
from .modules import SLAMCar


class MovingState(BaseState):
    def __init__(self, car: SLAMCar, events: list, id: int = 0) -> None:
        transitions: Dict[int, Dict[str, Any]] = {
            1: {'trigger': events[1], 'pre_transition': car.open_front_lights}, # stop sign
            2: {'trigger': events[2], 'pre_transition': car.halt_car}, # traffic light
            3: {'trigger': events[3], 'pre_transition': car.halt_car}, # terminate
        }
        super().__init__(transitions, id)

    def handle_action(self, car: SLAMCar, control_queue: DoubleBuffer) -> None:
        car.halt_car(halt_time=0.2)
        car.handle_control(control_queue)


class StopSignState(BaseState):
    def __init__(self, car: SLAMCar, events: list, id: int = 1) -> None:
        transitions: Dict[int, Dict[str, Any]] = {
            0: {'trigger': events[0], 'pre_transition': car.resume_car}, # moving
            3: {'trigger': events[3], 'pre_transition': car.halt_car}, # terminate
        }
        super().__init__(transitions, id)

    def handle_action(self, car: SLAMCar) -> None:
        car.handle_stop_sign()


class RedLightState(BaseState):
    def __init__(self, car: SLAMCar, events: list, id: int = 2) -> None:
        transitions: Dict[int, Dict[str, Any]] = {
            0: {'trigger': events[0], 'pre_transition': car.resume_car}, # moving
            3: {'trigger': events[3], 'pre_transition': car.halt_car}, # terminate
        }
        super().__init__(transitions, id)

    def handle_action(self, car: SLAMCar, control_queue: DoubleBuffer) -> None:
        car.halt_car(halt_time=0.2)


class FinalState(BaseState):
    def __init__(self, car: SLAMCar, id: int = 3) -> None:
        transitions: Dict[int, Dict[str, Any]] = {}
        super().__init__(transitions, id)

    def handle_action(self, car: SLAMCar, control_queue: DoubleBuffer) -> None:
        car.terminate()
