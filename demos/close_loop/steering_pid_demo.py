import time

import cv2

from demos.simulator_demo import simulator_demo
from .vehicle import VSPIDTestCar
from core.qcar.vehicle import PhysicalCar

def steering_pid_demo() -> None: 
    simulator_demo() # prepare the map
    car: PhysicalCar = VSPIDTestCar(0.3, 0.5)
    car.setup(-1.0163443448, -0.000, -0.19878977558)
    car.steering_control.start = time.time()
    try: 
        while True: 
            car.execute()
            # cv2.waitKey(1)
    except KeyboardInterrupt:
        # stop the car
        car.running_gear.read_write_std(0, 0)