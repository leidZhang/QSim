from typing import List, Callable

import numpy as np

from qvl.qlabs import QuanserInteractiveLabs
from pal.products.qcar import QCar
from core.qcar import QCAR_ACTOR_ID
from core.qcar.virtual import VirtualRuningGear

EGO_ID: int = 0

class RunningGear:
    def __init__(
        self, 
        actor_number: int, 
        qlabs: QuanserInteractiveLabs
    ) -> None:
        self.actor_number: int = actor_number
        self.qlabs: QuanserInteractiveLabs = qlabs
        self.read_write_std: Callable[[float, float, np.ndarray], None]
        self.__configure_running_gear(actor_number)

    def __configure_running_gear(self, actor_number: int) -> None:
        if actor_number != EGO_ID:
            self.running_gear = VirtualRuningGear(
                class_id=QCAR_ACTOR_ID,
                actor_number=actor_number,
            )
            self.read_write_std = self.__step_virtual_control
        else:
            self.running_gear = QCar()
            self.read_write_std = self.__step_qcar_control
        
    def __step_qcar_control(self, throttle: float, steering: float, led: List[int] = [0, 0, 0, 0, 0, 0, 0, 0]) -> None:
        self.running_gear.read_write_std(throttle, steering, led)

    def __step_virtual_control(self, throttle: float, steering: float, led: List[int] = [0, 0, 0, 0, 0]) -> None:
        self.running_gear.read_write_std(self.qlabs, throttle, steering, led)
