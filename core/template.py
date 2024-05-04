from abc import ABC, abstractmethod

class ServiceModule(ABC): 
    @abstractmethod 
    def terminate(self) -> None: 
        pass 
 
    def is_valid(self) -> bool: 
        return True 
    
    def execute(self) -> None: 
        pass 

class Controller(ServiceModule): 
    @abstractmethod 
    def normalize_throttle(self, x_axis_signal) -> float: 
        pass 

    @abstractmethod 
    def normalize_steering(self, y_axis_signal) -> float: 
        pass 

    def run_proccess(self) -> None: 
        try: 
            while True: 
                self.execute() 
        except Exception: 
            self.terminate() 