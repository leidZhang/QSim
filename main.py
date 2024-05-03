import time 

from pal.products.qcar import QCar

from simulator import QLabSimulator
from roadmap import ACCRoadMap

if __name__ == "__main__":
    roadmap: ACCRoadMap = ACCRoadMap()
    node_id: int = 24
    x_pos, y_pose, angle = roadmap.nodes[node_id].pose

    sim = QLabSimulator(dt=1/30)
    sim.render_map([x_pos, y_pose, angle])

    car: QCar = QCar()
    car.read_write_std(throttle=0.05, steering=0.0)
    time.sleep(1)
    step: int = 1
    while step <= 10000: 
        sim.reset_map()
        start: float = time.time()
        car: QCar = QCar()
        car.read_write_std(throttle=0.05, steering=0.0)
        time.sleep(1)
        state = sim.get_actor_state('car')
        print(f"x: {state[0]}, y: {state[1]}, yaw: {state[2]}, speed: {state[3]}m/s, rate of turn: {state[4]}, acceleration: {state[5]}m/s^2")
        print(f"Frequency: {1 / (time.time() - start)} Hz")
        step += 1
        print(f"Step: {step}")
    
    car.read_write_std(throttle=0.0, steering=0.0)
    print("Simulation complete")
    