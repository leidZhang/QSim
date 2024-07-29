from multiprocessing import Process

import cv2
import numpy as np

from qvl.qlabs import QuanserInteractiveLabs
from core.qcar.constants import QCAR_ACTOR_ID
from core.qcar.virtual import VirtualRuningGear
from core.roadmap import ACCRoadMap
from core.policies.pure_persuit import PurePursuiteAdaptor
from core.environment.simulator import QLabSimulator
from control.modules import WaypointCar


def run_car(id: int, waypoints: np.ndarray):
    qlabs: QuanserInteractiveLabs = QuanserInteractiveLabs()
    qlabs.open("localhost")
    car: WaypointCar = WaypointCar(id, 0.03, qlabs, 0.08, 0.5)
    if id != 0:
        car.set_running_gear(VirtualRuningGear(QCAR_ACTOR_ID, id, qlabs))

    policy = PurePursuiteAdaptor(max_lookahead_distance=0.5)
    car.setup(waypoints)

    step = 0
    while True:       
        observation = car.observation
        action, _ = policy.execute(observation)
        car.execute(action)
        step += 1


if __name__ == "__main__":
    roadmap = ACCRoadMap()
    task_1 = [10, 2, 4, 14, 20, 22, 10]
    task_2 = [5, 3, 13, 19, 17, 15, 5]
    waypoints_1 = roadmap.generate_path(task_1)
    waypoints_2 = roadmap.generate_path(task_2)    

    pos1 = roadmap.nodes[task_1[0]].pose
    pos2 = roadmap.nodes[task_2[0]].pose

    sim = QLabSimulator(offsets=(0, 0))
    print("Rendering map...")
    sim.render_map()
    print("Resetting car pos...")
    sim.reset_map([pos1[0], pos1[1], 0], [0, 0, pos1[2]])
    print("Adding new car...")
    sim.add_car([pos2[0], pos2[1], 0], [0, 0, pos2[2]])

    p1 = Process(target=run_car, args=(0, waypoints_1))
    p2 = Process(target=run_car, args=(1, waypoints_2))

    p1.start()
    p2.start()
    # car1 = QCar(id=0)
 
    # qlabs: QuanserInteractiveLabs = QuanserInteractiveLabs()
    # qlabs.open("localhost")
    # virtual_car = VirtualRuningGear(QCAR_ACTOR_ID, 1)
    # while True:
    #     virtual_car.read_write_std(qlabs, 0.08, 0.1)
    #     car1.read_write_std(0.08, 0.1)
