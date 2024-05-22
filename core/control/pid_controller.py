import time
from abc import ABC

import numpy as np

from pal.utilities.math import Filter


class PIDController(ABC): 
    """
    The PIDController class is an abstract class that implements the PID controller

    Attributes:
    - integral_error: float: The integral error of the PID controller
    - cross_error: float: The cross error of the PID controller
    - prev_cross_error: float: The previous cross error of the PID controller
    - prev_derivative_term: float: The previous derivative term of the PID controller
    - prev_state: float: The previous state of the PID controller
    - upper_bound: float: The upper bound of the PID controller
    - lower_bound: float: The lower bound of the PID controller
    - frequecy_filter: Filter: The frequency filter of the PID controller

    Methods:
    - setup: Sets up the PID controller
    - execute: Executes the PID controller
    """
    
    def __init__(self, upper_bound: float, lower_bound: float) -> None:
        """
        Initializes the PIDController object

        Parameters:
        - upper_bound: float: The upper bound of the PID controller
        - lower_bound: float: The lower bound of the PID controller

        Returns:
        - None
        """
        # error terms
        self.integral_error: float = 0.0
        self.cross_error: float = 0.0
        self.prev_cross_error: float = 0.0
        self.prev_derivative_term: float = 0.0
        # state terms
        self.prev_state: float = 0.0  
        self.upper_bound: float = upper_bound
        self.lower_bound: float = lower_bound
        # frequency filter
        self.frequecy_filter: Filter = Filter().low_pass_first_order_variable(90, 0.01)
        next(self.frequecy_filter)

    def setup(self, k_p:float, k_i: float, k_d: float) -> None:
        """
        The setup method to set the PID gains

        Parameters:
        - k_p: float: The proportional gain of the PID controller
        - k_i: float: The integral gain of the PID controller
        - k_d: float: The derivative gain of the PID controller

        Returns:
        - None
        """
        # gains
        self.k_p: float = k_p
        self.k_i: float = k_i
        self.k_d: float = k_d
        self.dt: float = 0.0
        self.start = time.time()

    def execute(self) -> float:
        """
        The base execute method to execute the PID controller, the calculation of 
        the cross error needs to be implemented by the child class

        Returns:
        - float: The state of the PID controller
        """
        self.dt = time.time() - self.start 
        control_rate: float = 1 / self.dt
        self.frequecy_filter = Filter().low_pass_first_order_variable(control_rate-5, self.dt, self.prev_state)
        next(self.frequecy_filter)

        self.start = time.time()
        self.integral_error += self.dt * self.cross_error
        derivetive_error: float = (self.cross_error - self.prev_cross_error) / self.dt
        raw_state: float = self.k_p * self.cross_error + self.k_i * self.integral_error + self.k_d * derivetive_error
        state: float = self.frequecy_filter.send((np.clip(raw_state, self.lower_bound, self.upper_bound), self.dt))
        self.prev_state = state # update the previous state
        # save the last cross error
        self.prev_cross_err = self.cross_error
        self.previous_derivative_term = derivetive_error

        return state
        