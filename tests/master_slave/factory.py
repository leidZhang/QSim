from threading import Event
from typing import Tuple, Dict, List, Any

from core.utils.inter_threads import BaseComm
from core.qcar.factory import CarFactory
from .modules import PIDControlCar, EdgeFinderComm, ObserveComm, CarComm
from .constants import DEFAULT_INTERCEPT_OFFSET, DEFAULT_SLOPE_OFFSET
from .constants import THROTTLE_DEFAULT_K_P, THROTTLE_DEFAULT_K_I, THROTTLE_DEFAULT_K_D
from .constants import STEERING_DEFAULT_K_P, STEERING_DEFAULT_K_I, STEERING_DEFAULT_K_D


class PIDControlCarFactory(CarFactory):
    def __init__(self) -> None:
        super().__init__()
        # the default settings of pid control car
        self.offsets: Tuple[float, float] = (DEFAULT_SLOPE_OFFSET, DEFAULT_INTERCEPT_OFFSET)
        self.pid_gains: Dict[str, List[float]] = {
            'steering': [STEERING_DEFAULT_K_P, STEERING_DEFAULT_K_I, STEERING_DEFAULT_K_D],
            'throttle': [THROTTLE_DEFAULT_K_P, THROTTLE_DEFAULT_K_I, THROTTLE_DEFAULT_K_D]
        }

    def _create_car(self) -> None:
        self.car = PIDControlCar(
            throttle_coeff=self.throttle_coeff, steering_coeff=self.steering_coeff
        )

    def _setup_car(self) -> None:
        self.car.setup(pid_gains=self.pid_gains, offsets=self.offsets)

    def set_offsets(self, offsets: Tuple[float, float]) -> None:
        self.offsets = offsets

    def set_pid_gains(self, pid_gains: Dict[str, List[float]]) -> None:
        # format check
        if 'steering' not in pid_gains.keys() or 'throttle' not in pid_gains.keys():
            raise ValueError("Invalid input, the input's key set must have 'steering' and 'throttle'")
        if len(pid_gains['steering']) != 3 or len(pid_gains['throttle']) != 3:
            raise ValueError("Invalid input format, the values should be list that have length 3")
        # set the pid gains as attribute
        self.pid_gains = pid_gains

    def build_car(self) -> PIDControlCar:
        self._create_car()
        self._setup_car()
        return self.car


class CommFactory:
    def create_comm_thread(self, type: str, core_instance: Any, event: Event) -> BaseComm:
        if type == "edge_finder_comm":
            return EdgeFinderComm(event=event, camera=core_instance)
        elif type == "observe_comm":
            return ObserveComm(event=event, camera=core_instance)
        elif type == "car_comm":
            car: PIDControlCar = core_instance.build_car()
            return CarComm(event=event, car=car)
        else:
            raise ValueError(f"Cannot build comm module for {type} type")
