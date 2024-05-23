import time

from demos.simulator_demo import simulator_demo

from .vehicle import ThrottlePIDTestCar
from core.qcar.vehicle import PhysicalCar
from core.utils.tools import plot_data_in_list

def throttle_pid_demo():
    simulator_demo() # prepare the map
    car: PhysicalCar = ThrottlePIDTestCar(0.3, 0.5)
    car.setup(k_p=0.204, k_i=1.85, k_d=0.0008) # 0.208
    car.throttle_control.start = time.time()
    time.sleep(0.03)
    throttle_history: list = []

    try: 
        while True:
            car.execute()
            throttle: float = round(car.throttle, 4)
            throttle_history.append(throttle)
    except KeyboardInterrupt:
        plot_data_in_list(throttle_history, "Throttle PID Test", "Time", "PWM")