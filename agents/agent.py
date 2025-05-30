from typing import List, Any, Dict, Union

import numpy as np
from shapely.geometry import Point, Polygon
from qvl.qlabs import QuanserInteractiveLabs

from core.control import WaypointProcessor
from core.templates import PolicyAdapter, BasePolicy
from .fsm import FSM
from .interfaces import RunningGear

class CarAgent:
    def __init__(self, id: int, qlabs: QuanserInteractiveLabs) -> None:
        self.id: int = id
        self.speed_lock: bool = False
        self.observation: Dict[str, Any] = {} 
        self.passed_departure_area: bool = False
        self.is_hazard: bool = False

        self.policy: Union[BasePolicy, PolicyAdapter] = None
        self.running_gear: RunningGear = RunningGear(id, qlabs)
        self.fsm: FSM = FSM(self.running_gear)
        self.segmentor: WaypointProcessor = WaypointProcessor(auto_stop=True)
        self.pending_area: Polygon = None
        self.departure_area: Polygon = None
        
    def reset(
        self, 
        waypoints: np.ndarray, 
        agent_states: List[np.ndarray],
        init_index: int = 0
    ) -> None:
        self.observation = {}
        self.passed_departure_area = False
        self.fsm.reset()

        state: np.ndarray = agent_states[self.id]
        self.observation = self.segmentor.reset(
            ego_state=state,
            observation=self.observation,
            waypoints=waypoints,
            init_waypoint_index=init_index
        )
        self.observation['action'] = np.zeros(2)

    def step(self, agent_states: List[np.ndarray]) -> None:
        state: np.ndarray = agent_states[self.id]
        self.observation = self.segmentor.step(state, self.observation)
        action, _ = self.policy.execute(self.observation)
        self.observation['action'] = action
        self.fsm.step(self.observation, self.pending_area, self.speed_lock)

    def has_passed_crossroads(self) -> bool:
        if not self.passed_departure_area:
            point: Point = Point(self.observation['state_info'][:2])
            if self.departure_area.contains(point):
                self.passed_departure_area = True
                self.is_hazard = False # If hazard exit the crossroads, it wont harm the ego
                print(f"Agent {self.id} has passed crossroads")
        return self.passed_departure_area
    
    def get_waypoints(self) -> np.ndarray:
        return self.segmentor.waypoints
    
    def get_state(self) -> int:
        return self.fsm.current_state_id

    def set_speed_lock(self, lock: bool) -> None:
        self.speed_lock = lock

    def set_policy(self, policy: Union[BasePolicy, PolicyAdapter]) -> None:
        self.policy = policy

    def set_pending_area(self, area: Polygon) -> None:
        self.pending_area = area

    def set_departure_area(self, area: Polygon) -> None:
        self.departure_area = area

    def set_init_state(self, state_id: int) -> None:
        self.fsm.set_init_state(state_id)

    def terminate(self) -> None:
        self.running_gear.read_write_std(0, 0)
