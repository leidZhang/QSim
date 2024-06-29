from typing import Tuple, List

import numpy as np
from qvl.qlabs import QuanserInteractiveLabs

from core.roadmap import ACCRoadMap
from core.environment import QLabEnvironment


class PerformanceTestEnvironment(QLabEnvironment): # demo environment will do nothing
    def handle_reward(self, *args) -> Tuple[float, bool]:
        return 0.0, False
    
    def step(self, action: np.ndarray, metrics: np.ndarray) -> Tuple[dict, float, bool, dict]:
        return None, 0.0, False, None
    
    def reset(self, location: list, orientation: list) -> Tuple[dict, float, bool, dict]:
        return super().reset(location=location, orientation=orientation)
    
def destroy_map() -> int:
    qlabs = QuanserInteractiveLabs()
    qlabs.open('localhost')
    qlabs.destroy_all_spawned_actors()
    qlabs.close()

def prepare_test_environment(node_id: int = 10) -> np.ndarray:
    roadmap: ACCRoadMap = ACCRoadMap()
    x_pos, y_pose, angle = roadmap.nodes[node_id].pose
    waypoint_sequence = roadmap.generate_path([10, 2, 4, 14, 20])
    simulator: QLabEnvironment = PerformanceTestEnvironment(dt=0.05, privileged=True)
    simulator.setup(nodes=None, sequence=waypoint_sequence)
    simulator.reset(location=[x_pos, y_pose, 0], orientation=[0, 0, angle])

    return waypoint_sequence

def prepare_test_environment_waypoint(waypoint_index: int = 0) -> None:
    roadmap: ACCRoadMap = ACCRoadMap()
    waypoint_sequence = roadmap.generate_path([10, 4, 14, 20, 22, 10])

    simulator: QLabEnvironment = PerformanceTestEnvironment(dt=0.05, privileged=True)
    simulator.setup(nodes=None, sequence=waypoint_sequence)
    location, orientation = simulator.spawn_on_waypoints(waypoint_index=waypoint_index)
    simulator.reset(location=location, orientation=orientation)

    return waypoint_sequence