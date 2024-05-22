from core.control.pid_controller import PIDController
from .constants import DEFAULT_SLOPE_OFFSET, DEFAULT_INTERCEPT_OFFSET


class SteeringPIDController(PIDController): 
    """
    The SteeringPIDController class is a class that generates the steering angle for the car
    based on the slope and intercept on the frame 

    Attributes:
    - slope_offset: float: The slope offset of the lane
    - intercept_offset: float: The intercept offset of the lane
    """

    def __init__(self, upper_bound: float, lower_bound: float) -> None:
        """
        Initializes the SteeringPIDController object

        Parameters:
        - upper_bound: float: The upper bound of the PID controller
        - lower_bound: float: The lower bound of the PID controller
        
        Returns:
        - None
        """
        super().__init__(upper_bound, lower_bound)
        self.slope_offset: float = DEFAULT_SLOPE_OFFSET
        self.intercept_offset: float = DEFAULT_INTERCEPT_OFFSET

    def execute(self, input: tuple, image_width: float) -> float:
        """
        The execute method to generate the steering angle based on the slope and intercept

        Parameters:
        - input: tuple: The input of the steering PID controller
        - image_width: float: The width of the image

        Returns:
        - float: The steering angle of the car
        """
        # decode the input
        slope: float = input[0]
        intercept: float = input[1]

        # fault tolerance
        if slope == 0.3419: return 0.0
        if abs(slope) < 0.2 and abs(intercept) < 100:
            slope = self.slope_offset
            intercept = self.intercept_offset
        # calculate the cross error
        self.cross_error: float = (intercept/-slope) - (self.intercept_offset / -self.slope_offset)
        self.cross_err = self.cross_err / image_width
        # calculate the steering angle
        steering: float = super().execute()

        return steering