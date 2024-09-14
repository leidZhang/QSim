# from typing import Dict, List

# import cv2
# import numpy as np

# from core.roadmap import ACCRoadMap
# from core.roadmap.raster_map import ACCRasterMap
# from generator.env_raster_map import CREnvRasterMap


# if __name__ == '__main__':
#     map_params: dict = {
#         "lanes": ((255, 255, 255), 1),
#         "hazards": ((255, 0, 255), 2),
#         "waypoints": ((255, 255, 0), 3)
#     }
#     map_size: tuple = (288, 288, 3)
#     ego_state: np.ndarray = np.array([0.15, 0.950, np.pi, 0.0, 0.0, 0.0])
#     hazard_states: List[float] = [
#         [0.0, 2.118, -np.pi / 2, 0.0, 0.0, 0.0],
#         [0.264, -0.29, np.pi / 2, 0.0, 0.0, 0.0],
#         [1.206, 1.082, np.pi, 0.0, 0.0, 0.0],
#         [-1.036, 0.816, 0.0, 0.0, 0.0, 0.0],
#     ]
#     routes: List[List[int]] = [
#         [12, 8, 10],
#         [1, 13, 19, 17],
#         [6, 0, 2],
#         [9, 0, 2],
#     ]
#     waypoints_list: List[np.ndarray] = []

#     roadmap: ACCRoadMap = ACCRoadMap()
#     renderer: CREnvRasterMap = CREnvRasterMap(roadmap, map_size, map_params)
#     for i, route in enumerate(routes):
#         waypoints = roadmap.generate_path(route)
#         waypoints_list.append(waypoints[:150])

#     raster_map, _, _ = renderer.draw_map(ego_state, hazard_states, waypoints_list)
#     cv2.imshow("Raster Map", raster_map)
#     cv2.waitKey(0)
