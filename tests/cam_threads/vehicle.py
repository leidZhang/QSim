from typing import Tuple
from core.qcar import PhysicalCar


# Testbed is only used to forward and reverse periodically
class TestBed(PhysicalCar): 
    def execute(self, pwm: float) -> Tuple:
        self.running_gear.read_write_std(throttle=pwm, steering=0.0)