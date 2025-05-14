from abc import ABC
from typing import List, Dict, Any

import cv2
import numpy as np

from hal.utilities.path_planning import RoadMap
from .constants import PIXEL_PER_METER, ACC_MAP_PARAMS
from .constants import TRAFFIC_LIGHT_COLORS, TRAFFIC_LIGHT_SIGNALS


def get_lane_edge_polylines(path_waypoints: np.ndarray, road_width: float) -> tuple:
    path_polyline: np.ndarray = np.transpose(path_waypoints)
    dw: np.ndarray = np.diff(path_polyline, axis=0)
    theta: np.ndarray = np.arctan2(dw[:, 1], dw[:, 0])
    theta: np.ndarray = np.concatenate([[theta[0]], theta])

    right_lane: np.ndarray = path_polyline + np.transpose(np.array([np.cos(theta + (np.pi/2)), \
        np.sin(theta + (np.pi/2))])) * (road_width / 2)
    left_lane: np.ndarray = path_polyline + np.transpose(np.array([np.cos(theta - (np.pi/2)), \
        np.sin(theta - (np.pi/2))])) * (road_width / 2)

    return np.transpose(right_lane[::2]), np.transpose(left_lane[::2])


def to_pixel(
    pose: np.ndarray,
    state: np.ndarray,
    offsets: float = (1.0, 0.4),
    ratial: float = PIXEL_PER_METER,
    map_size: int = 192,
    fixed: bool = False
) -> np.ndarray:
    x, y, yaw = state
    yaw -= np.pi / 2
 
    if not fixed:
        rot: np.ndarray = np.array([
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw), np.cos(yaw)],
        ])        

        pose = pose - np.array([x, y])
        pose = np.matmul(pose, rot)
        # pose = pose + np.array([x, y])

        top_left_px: np.ndarray = np.array([-map_size / 2, -map_size / 2])  
    else:
        top_left_px: np.ndarray = np.array([
            -offsets[0] - 0.45, -2.5 - offsets[1]
        ], dtype=np.float32) * ratial 

    pose = pose * PIXEL_PER_METER
    pose[..., 0] = pose[..., 0] - top_left_px[0]
    pose[..., 1] = -pose[..., 1] - top_left_px[1]

    return np.round(pose).astype(np.int32)


def get_bounding_box_poly(box: np.ndarray, state: np.ndarray) -> np.ndarray:
    x, y, yaw = state
    rot: np.ndarray = np.array([
        [np.cos(yaw), np.sin(yaw)],
        [-np.sin(yaw), np.cos(yaw)],
    ])
    return np.array([x, y]) + np.matmul(box, rot)


def get_polyline_layer(
    layer: np.ndarray,
    waypoints: np.ndarray,
    state: np.ndarray,
    rgb_value: tuple,
    offsets: float = (1.0, 0.4),
    ratial: float = PIXEL_PER_METER
) -> np.ndarray:
    polyline: np.ndarray = np.transpose(waypoints)
    cv2.polylines(
        layer,
        [to_pixel(polyline, state, offsets=offsets, ratial=ratial)],
        color=rgb_value,
        thickness=1,
        isClosed=False,
        lineType=cv2.LINE_AA
    )
    return layer


def get_box_layer(
    layer: np.ndarray,
    box_poly: np.ndarray,
    state: np.ndarray,
    rgb_value: tuple,
    offsets: tuple = (1.0, 0.4),
    ratial: float = PIXEL_PER_METER
) -> np.ndarray:
    cv2.fillPoly(
        layer,
        [to_pixel(box_poly, state, offsets=offsets, ratial=ratial)],
        color=rgb_value
    )
    return layer


class RasterMapRenderer(ABC):
    def __init__(
        self,
        road_map: RoadMap,
        map_size: tuple = (192, 192, 3),
        map_params: Dict[str, tuple] = ACC_MAP_PARAMS
    ) -> None:
        # create polylines for rendering
        self.agent_length: float = 0.4
        self.agent_width: float = 0.19
        self.road_width: float = 0.27
        self.map_size: tuple = map_size
        self.map_params: Dict[str, tuple] = map_params
        self.map_polylines: List[np.ndarray] = []
        self.__add_map_polylines(road_map)
        self.bounding_box: np.ndarray = np.array([
            [-(self.agent_length / 2), -(self.agent_width / 2)],
            [+(self.agent_length / 2), -(self.agent_width / 2)],
            [+(self.agent_length / 2), +(self.agent_width / 2)],
            [-(self.agent_length / 2), +(self.agent_width / 2)],
        ])

    def __add_map_polylines(self, road_map: RoadMap) -> None:
        for edge in road_map.edges:
            right_lane, left_lane = get_lane_edge_polylines(edge.waypoints, 0.27)
            self.map_polylines.append(right_lane)
            self.map_polylines.append(left_lane)

    def seg_to_image(self, seg: np.ndarray) -> np.ndarray:
        raster_map = np.zeros(self.map_size, dtype=np.uint8)
        for k in self.map_params:
            indices = seg == self.map_params[k][1]
            raster_map[indices] = self.map_params[k][0]
        return raster_map

    def draw_map(self, *args) -> tuple:
        raster_map = np.zeros(self.map_size, dtype=np.uint8)
        segmentation_target: np.ndarray = np.zeros(self.map_size[:2], dtype=np.uint8)
        map_info: Dict[str, np.ndarray] = {
            k: np.zeros(self.map_size[:2], dtype=bool) for k in self.map_params.keys()
        }
        return raster_map, segmentation_target, map_info


class ACCRasterMap(RasterMapRenderer):
    def __init__(self, road_map: RoadMap, map_params: Dict[str, Any] = ACC_MAP_PARAMS) -> None:
        super().__init__(road_map)
        self.map_params: Dict[str, Any] = map_params

    def _draw_ego_agent(self, pose: np.ndarray, map_info: Dict[str, Any]) -> None:
        ego_agent_layer: np.ndarray = np.zeros(self.map_size, dtype=np.uint8)
        ego_box: np.ndarray = get_bounding_box_poly(self.bounding_box, pose)
        ego_agent_layer = get_box_layer(ego_agent_layer, ego_box, pose, (255, 0, 0))
        map_info["ego"] = np.all(ego_agent_layer == (255, 0, 0), axis=-1)

    def _draw_edge_layer(self, pose: np.ndarray, map_info: Dict[str, Any]) -> None:
        edge_layer: np.ndarray = np.zeros(self.map_size, dtype=np.uint8)
        for map_polyline in self.map_polylines:
            edge_layer = get_polyline_layer(edge_layer, map_polyline, pose, (255, 255, 255))
        map_info["lanes"] = edge_layer[:, :, 0] > 0

    def _draw_traffic_lights_layer(self, pose: np.ndarray, map_objects: dict, map_info: Dict[str, Any]) -> None:
        traffic_light_layer: np.ndarray = np.zeros(self.map_size, dtype=np.uint8)
        for i, traffic_light in map_objects["traffic_lights"].items():
            color: tuple = TRAFFIC_LIGHT_COLORS[traffic_light.state]
            traffic_light_layer = get_box_layer(traffic_light_layer, traffic_light.bounding_box, pose, color)
        map_info[f"green_lights"] = np.all(traffic_light_layer == TRAFFIC_LIGHT_COLORS[0], axis=-1)
        map_info[f"yellow_lights"] = np.all(traffic_light_layer == TRAFFIC_LIGHT_COLORS[1], axis=-1)
        map_info[f"red_lights"] = np.all(traffic_light_layer == TRAFFIC_LIGHT_COLORS[2], axis=-1)

    def _draw_objects_layer(self, pose: np.ndarray, agents: list, map_info: Dict[str, Any]) -> None:
        agents_layer: np.ndarray = np.zeros(self.map_size, dtype=np.uint8)
        for agent in agents:
            agent_state: np.ndarray = agent[:3]
            agent_box: np.ndarray = get_bounding_box_poly(self.bounding_box, agent_state)
            agents_layer = get_box_layer(agents_layer, agent_box, pose, (255, 0, 255))
        map_info["objects"] = np.all(agents_layer == (255, 0, 255), axis=-1)

    def draw_map(self, state: np.ndarray, agents: list, map_objects: dict) -> tuple:
        pose: np.ndarray = state[:3] # we do not need velocity and acceleration for rendering
        raster_map, segmentation_target, map_info = super().draw_map() # draw base map

        self._draw_ego_agent(pose, map_info)
        self._draw_edge_layer(pose, map_info)
        self._draw_traffic_lights_layer(pose, map_objects, map_info)
        self._draw_objects_layer(pose, agents, map_info)

        # render raster map
        for k in map_info:
            raster_map[map_info[k]] = self.map_params[k][0]
            segmentation_target[map_info[k]] = self.map_params[k][1]
        return raster_map, segmentation_target, map_info
