class HaltException(Exception): 
    """
    Exception raised when the car is required to stop. It will also specify the time for which 
    the car should stop.

    Attributes:
        stop_time (float): The time for which the car should stop.
    """
    
    def __init__(self, stop_time: float = 3.0, message: str = '', *args: object) -> None: 
        super().__init__(*args)
        self.message: str = message
        self.stop_time: float = stop_time


class StopException(Exception):
    """
    Exception raised when the car is required to stop. It will also specify the time for which 
    the car should stop.

    Attributes:
        stop_time (float): The time for which the car should stop.
    """
    
    def __init__(self, *args: object) -> None: 
        super().__init__(*args)