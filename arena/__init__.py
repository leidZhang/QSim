from typing import Tuple, List

from core.roadmap import ACCRoadMap
from core.environment.simulator import QLabSimulator
from system.settings import EGO_VEHICLE_TASK, BOT_TASKS
from .modules import RealWorldEnv


def get_pose(roadmap: ACCRoadMap, start_node: int) -> Tuple[List[float], List[float]]:
    x_pos, y_pose, angle = roadmap.nodes[start_node].pose
    location: List[float] = [x_pos, y_pose, 0]
    orientation: List[float] = [0, 0, angle]
    return location, orientation


roadmap: ACCRoadMap = ACCRoadMap()    
sim: QLabSimulator = QLabSimulator((0, 0))
poses: List[Tuple[List[float], List[float]]] = [get_pose(roadmap, EGO_VEHICLE_TASK[0])]
for task in BOT_TASKS:
    poses.append(get_pose(roadmap, task[0]))
env: RealWorldEnv = RealWorldEnv(roadmap, sim, poses)
