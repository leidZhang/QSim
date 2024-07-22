import numpy as np

from core.roadmap import ACCRoadMap

def prepare_real_world_task(node_sequence: np.ndarray) -> np.ndarray:
    roadmap: ACCRoadMap = ACCRoadMap()
    return roadmap.generate_path(node_sequence)