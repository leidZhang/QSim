from core.control.pid_controller import PIDController

class ThrottlePIDController(PIDController): 
    def __init__(self, upper_bound: float, lower_bound: float) -> None:
        super().__init__(upper_bound, lower_bound)
        # implement the __init__ method

    def execute(self, linear_speed: float) -> float:
        # implement the execute method
        return super().execute()