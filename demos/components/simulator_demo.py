from typing import Tuple, List

from numpy import ndarray
from qvl.qlabs import QuanserInteractiveLabs

from core.roadmap import ACCRoadMap
from core.environment import QLabEnvironment


class DemoEnvironment(QLabEnvironment): # demo environment will do nothing
    def handle_reward(self, *args) -> Tuple[float, bool]:
        return 0.0, False
    
    def step(self, action: ndarray, metrics: ndarray) -> Tuple[dict, float, bool, dict]:
        return None, 0.0, False, None
    
    def reset(self, location: list, orientation: list) -> Tuple[dict, float, bool, dict]:
        return super().reset(location=location, orientation=orientation)

def simulator_demo(node_id: int = 4) -> None:
    roadmap: ACCRoadMap = ACCRoadMap()
    qlabs = QuanserInteractiveLabs()
    x_pos, y_pose, angle = roadmap.nodes[node_id].pose
    waypoint_sequence = roadmap.generate_path([4, 14, 20, 22, 10])

    simulator: QLabEnvironment = DemoEnvironment(dt=0.05, privileged=True)
    simulator.setup(nodes=None, sequence=waypoint_sequence)
    simulator.reset(location=[x_pos, y_pose, 0], orientation=[0, 0, angle])
    