from typing import List, Any

import cv2
import numpy as np

from core.roadmap.constants import *
from core.roadmap.raster_map import *


CROSS_ROARD_RATIAL: float = 384 * 0.24
OFFSET: tuple = (1.55, 0.75)
CR_MAP_SIZE: tuple = (384, 384, 3)
CR_MAP_PARAMS: dict = {
    "lanes": ((255, 255, 255), 1),
    "hazards": ((255, 0, 255), 2),
    "waypoints": ((255, 255, 0), 3)
}


class HITLRasterMap(RasterMapRenderer):
    def _draw_edge_layer(self, pose: np.ndarray, map_info: Dict[str, Any]) -> None:
        edge_layer: np.ndarray = np.zeros(self.map_size, dtype=np.uint8)
        for map_polyline in self.map_polylines:
            edge_layer = get_polyline_layer(edge_layer, map_polyline, pose, (255, 255, 255), OFFSET, CROSS_ROARD_RATIAL)
        map_info["lanes"] = edge_layer[:, :, 0] > 0

    def _draw_waypoints_layer(self, pose: np.ndarray, waypoints_list: List[np.ndarray], map_info: Dict[str, Any]) -> None:
        waypoints_layer: np.ndarray = np.zeros(self.map_size, dtype=np.uint8)
        for waypoints in waypoints_list:
            waypoint_polyline: np.ndarray = np.transpose(waypoints)
            waypoints_layer: np.ndarray = get_polyline_layer(
                waypoints_layer, waypoint_polyline, pose, (255, 255, 0), OFFSET, CROSS_ROARD_RATIAL
            )
        map_info["waypoints"] = waypoints_layer[:, :, 0] > 0

    def _draw_hazard_layer(self, pose: np.ndarray, hazards: list, map_info: Dict[str, Any]) -> None:
        hazards_layer: np.ndarray = np.zeros(self.map_size, dtype=np.uint8)
        for hazard in hazards:
            hazard_state: np.ndarray = hazard[:3]
            hazard_box: np.ndarray = get_bounding_box_poly(self.bounding_box, hazard_state)
            hazards_layer = get_box_layer(hazards_layer, hazard_box, pose, (255, 0, 255), OFFSET, CROSS_ROARD_RATIAL)
        map_info["hazards"] = np.all(hazards_layer == (255, 0, 255), axis=-1)

    def render_map(self, raster_map: np.ndarray, map_info: Dict[str, Any]) -> np.ndarray:
        for k in map_info:
            raster_map[map_info[k]] = self.map_params[k][0]
        return raster_map

    def draw_map(self, state: np.ndarray, hazards: list, waypoint_lists: List[np.ndarray]) -> tuple:
        pose: np.ndarray = state[:3] # we do not need velocity and acceleration for rendering
        raster_map, segmentation_target, map_info = super().draw_map() # draw base map

        self._draw_hazard_layer(pose, hazards, map_info)
        self._draw_waypoints_layer(pose, waypoint_lists, map_info)

        # render raster map
        raster_map = self.render_map(raster_map, map_info)
        return raster_map, segmentation_target, map_info