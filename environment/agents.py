from abc import ABC
import time
from typing import List, Any, Dict, Union

import numpy as np

from qvl.qlabs import QuanserInteractiveLabs
from core.templates import PolicyAdapter, BasePolicy
from core.qcar import PhysicalCar
from core.qcar.virtual import VirtualRuningGear
from core.control import WaypointProcessor
from .utils import is_in_area_aabb, is_waypoint_intersected

class BaseAgent(ABC):
    def __init__(self, id: int) -> None:
        self.id: int = id
        self.speed_lock: bool = False
        self.normal: bool = True
        self.passed: bool = False

        self.policy: Union[BasePolicy, PolicyAdapter] = None
        self.observation: Dict[str, Any] = None 
        self.area: dict = None # bounding box area
    
    def reset(self) -> None:
        self.passed = False

    def step(self) -> None:
        ...

    def set_policy(self, policy: BasePolicy) -> None:
        self.policy = policy

    def set_area(self, area: dict) -> None:
        self.area = area

    def set_speed_lock(self, lock: bool) -> None:
        self.speed_lock = lock

    def has_passed_area(self) -> bool:
        if not self.passed:
            self.passed: bool = is_in_area_aabb(self.observation['state_info'], self.area)
        if self.passed:
            print(f"Agent {self.id} has passed the area")
        return self.passed


class CarAgent(BaseAgent, PhysicalCar):
    def __init__(self, id: int = 0, max_pwm: float = 0.08, qlabs: QuanserInteractiveLabs = None) -> None:
        BaseAgent.__init__(self, id)
        PhysicalCar.__init__(self)
        self.max_pwm: float = max_pwm
        self.qlabs: QuanserInteractiveLabs = qlabs
        self.preporcessor: WaypointProcessor = WaypointProcessor(auto_stop=True)
        self.observation: Dict[str, np.ndarray] = {"action": np.zeros(2)}

    def handle_action(self, action: np.ndarray) -> None:
        action[0] = action[0] * self.max_pwm
        throttle: float = action[0]
        steering: float = action[1] 
        if self.id == 0:
            self.running_gear.read_write_std(throttle, steering, self.leds)
        else:
            self.normal, _, _, _, _ = self.running_gear.read_write_std(self.qlabs, throttle, steering, self.leds)
        self.observation["action"] = action

    def reset(self, waypionts: np.ndarray, agent_states: List[np.ndarray]) -> None:
        super().reset()
        self._get_ego_state(agent_states)
        self.observation = self.preporcessor.setup(
            ego_state=self.observation["state_info"],
            observation=self.observation,
            waypoints=waypionts,
        )
        current_wayppint_index: int = self.preporcessor.current_waypoint_index
        start_waypoint_index: int = max(0, current_wayppint_index - 25)
        end_waypoint_index: int = min(len(waypionts) - 1, current_wayppint_index + 600)
        self.observation["global_waypoints"] = waypionts[
            start_waypoint_index:end_waypoint_index
        ]
        self.observation["progress"] = current_wayppint_index

    def step(self, agent_states: List[np.ndarray]) -> None:
        self._get_ego_state(agent_states)
        # self.handle_avoid_collide(agent_states)
        self._handle_preprocess()
        action, _ = self.policy.step(self.observation)
        if self.speed_lock or self.observation["done"]:
            self.handle_action([0, 0])
            return         
        self.handle_action(action)

    def set_running_gear(self, running_gear: VirtualRuningGear) -> None:
        self.running_gear = running_gear

    def handle_avoid_collide(self, agent_states: np.ndarray) -> None:
        look_ahead_traj: np.ndarray = self.observation['waypoints']
        for i, hazard_state in enumerate(agent_states):
            if i == self.id:
                continue

            orig: np.ndarray = self.observation['state_info'][:2]
            yaw: float = -self.observation['state_info'][2] + np.pi
            rot: np.ndarray = np.array([
                [np.cos(yaw), np.sin(yaw)],
                [-np.sin(yaw), np.cos(yaw)]
            ])
            hazard_traj: np.ndarray = np.matmul(np.array([hazard_state[:2]]) - orig, rot)
            if is_waypoint_intersected(look_ahead_traj, hazard_traj):
                print(f"Agent {self.id} is going to collide with agent {i}")
                self.observation["done"] = True
                return

    def _get_ego_state(self, agent_states: List[np.ndarray]) -> None:
        ego_state: np.ndarray = agent_states[self.id]
        hazard_states: List[np.ndarray] = [
            state if i != self.id and i != 0 else None for i, state in enumerate(agent_states)
        ]

        self.observation["state_info"] = ego_state
        self.observation["hazard_states"] = hazard_states

    def _handle_preprocess(self) -> None:
        self.observation = self.preporcessor.execute(self.observation["state_info"], self.observation)

    def get_waypoints(self) -> np.ndarray:
        return self.preporcessor.waypoints
    
    def halt_car(self, steering = 0, halt_time = 0.0):
        if self.id == 0:
            return super().halt_car(steering, halt_time)
        else:
            self.normal, _, _, _, _ = self.running_gear.read_write_std(self.qlabs, 0.0, steering, self.leds)
            time.sleep(halt_time)