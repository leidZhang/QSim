from abc import ABC
from .vehicle import BaseCar


class CarFactory(ABC):
    def __init__(self) -> None:
        self.car: BaseCar = None
        self.throttle_coeff: float = 0.3
        self.steering_coeff: float = 0.5

    def set_throttle_coeff(self, throttle_coeff: float) -> None:
        if throttle_coeff <= 0:
            raise ValueError('throttle_coeff cannot be smaller than 0')
        self.throttle_coeff = throttle_coeff

    def set_steering_coeff(self, steering_coeff: float) -> None:
        if steering_coeff <= 0:
            raise ValueError('steering_coeff cannot be smaller than 0')
        self.steering_coeff = steering_coeff