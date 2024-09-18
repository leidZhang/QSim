import time
from abc import abstractmethod
from typing import List, Tuple, Dict, Union

import numpy as np

from qvl.qlabs import QuanserInteractiveLabs
from core.qcar import CSICamera
from core.qcar import PhysicalCar
from core.qcar import VirtualOptitrack, QCAR_ACTOR_ID
from core.qcar.virtual import VirtualRuningGear
from core.templates import PolicyAdapter, BasePolicy
from core.control import WaypointProcessor
from .hazard_decision import HazardDetector


class StateDataBus: # get all the state information of the car agents
    def __init__(
        self,
        qlabs: QuanserInteractiveLabs,
        agent_list: list,
        dt: float
    ) -> None:
        self.qlabs: QuanserInteractiveLabs = qlabs
        self.clinets: List[VirtualOptitrack] = [
            VirtualOptitrack(QCAR_ACTOR_ID, id, dt) for id in range(len(agent_list))
        ]

    def reset(self) -> List[np.ndarray]:
        agent_states: List[np.ndarray] = self.step()
        for state in agent_states:
            state[-3:] = 0 # initial velocity and acceleration are always 0
        return agent_states

    def step(self) -> List[np.ndarray]:
        agent_states: List[np.ndarray] = []
        for client in self.clinets:
            client.read_state(self.qlabs)
            agent_states.append(client.state)
        return agent_states


class CarAgent(PhysicalCar): # ego vehicle, can be controlled by the user
    def __init__(self, actor_id: int = 0) -> None:
        super().__init__(throttle_coeff=0.10)
        self.actor_id: int = actor_id
        self.policy: Union[BasePolicy, PolicyAdapter] = None
        self.preporcessor: WaypointProcessor = WaypointProcessor(auto_stop=True)
        self.observation: Dict[str, np.ndarray] = {"action": np.zeros(2), "hazard_coeff": 1}

    def _get_ego_state(self, agent_states: List[np.ndarray], agent_ranks: List[int] = [0, 1, 2, 3]) -> None:
        ego_state: np.ndarray = agent_states[self.actor_id]
        hazard_states: List[np.ndarray] = [
            state if i != self.actor_id and i != 0 else None for i, state in enumerate(agent_states)
        ]
        self.hazard_ranks: List[int] = [
            rank if rank != self.actor_id and rank != 0 else None for rank in agent_ranks
        ]

        self.state = ego_state
        self.observation["hazards"] = hazard_states

    def reset(self, waypionts: np.ndarray, agent_states: List[np.ndarray]) -> None:
        self._get_ego_state(agent_states)
        self.observation = self.preporcessor.setup(
            ego_state=self.state,
            observation=self.observation,
            waypoints=waypionts,
        )
        current_wayppint_index: int = self.preporcessor.current_waypoint_index
        start_waypoint_index: int = max(0, current_wayppint_index - 30)
        end_waypoint_index: int = min(len(waypionts) - 1, current_wayppint_index + 600)
        self.observation["global_waypoints"] = waypionts[
            start_waypoint_index:end_waypoint_index
        ]
        self.observation["progress"] = current_wayppint_index # / len(waypionts)
        # self.observation["progress"] = np.linalg.norm(self.state[:2] - waypionts[-1])

    def _handle_preprocess(self) -> None:
        self.observation = self.preporcessor.execute(self.state, self.observation)
        current_wayppint_index: int = self.preporcessor.current_waypoint_index
        start_waypoint_index: int = max(0, current_wayppint_index - 30)
        end_waypoint_index: int = min(len(self.preporcessor.waypoints) - 1, current_wayppint_index + 180)
        self.observation["global_waypoints"] = self.preporcessor.waypoints[
            start_waypoint_index:end_waypoint_index
        ]

        self.observation["progress"] = current_wayppint_index # / len(self.preporcessor.waypoints)
        # self.observation["progress"] = np.linalg.norm(self.state[:2] - self.preporcessor.waypoints[-1])


    @abstractmethod
    def handle_action(self, action: np.ndarray) -> None:
        ...

    @abstractmethod
    def step(self, *args) -> None:
        ...

    def set_policy(self, policy: Union[BasePolicy, PolicyAdapter]) -> None:
        self.policy = policy

    def set_throttle_coeff(self, coeff: float) -> None:
        self.throttle_coeff = coeff

    def set_steering_coeff(self, coeff: float) -> None:
        self.steering_coeff = coeff


class EgoAgent(CarAgent):
    def __init__(self, actor_id: int = 0) -> None:
        super().__init__(actor_id)
        self.cameras: List[CSICamera] = [CSICamera(i) for i in range(4)]

    def __get_image_data(self) -> None:
        images: List[np.ndarray] = [None, None, None, None]
        for i, camera in enumerate(self.cameras):
            images[i] = camera.await_image()
        self.observation["images"] = images

    def handle_action(self, action: np.ndarray) -> None:
        throttle: float = 0 # action[0] * self.throttle_coeff
        steering: float = action[1] * self.steering_coeff
        self.running_gear.read_write_std(throttle, steering)
        self.observation["action"] = action

    def reset(self, waypoints: np.ndarray, agent_states: List[np.ndarray]) -> None:
        self.__get_image_data()
        super().reset(waypoints, agent_states)

    def step(self, agent_states: List[np.ndarray], *args) -> None:
        self.__get_image_data()
        self._get_ego_state(agent_states)
        self._handle_preprocess()
        action, _ = self.policy.execute(self.observation)
        self.handle_action(action)


class HazardAgent(CarAgent):
    def __init__(self, actor_id: int, qlabs: QuanserInteractiveLabs) -> None:
        super().__init__(actor_id)
        self.qlabs: QuanserInteractiveLabs = qlabs
        self.running_gear: VirtualRuningGear = VirtualRuningGear(QCAR_ACTOR_ID, actor_id)
        self.detector: HazardDetector = HazardDetector()

    def halt_car(self, steering: float = 0, halt_time: float = 0.1) -> None:
        self.running_gear.read_write_std(self.qlabs, throttle=0.0, steering=steering)
        time.sleep(halt_time)

    def handle_avoid_collide(self, agent_trajs: np.ndarray, agent_progresses: np.ndarray) -> None:
        decision: int = 1
        for i, traj in enumerate(agent_trajs):
            if i == 0 or i == self.actor_id:
                continue            
            print(f"Agent {self.actor_id} checking agent {i}'s trajectory...")
            if decision == 0:
                print(f"Agent {self.actor_id} detects agent {i} as a hazard")
                break            

            ego_traj: np.ndarray = self.observation["global_waypoints"]
            if agent_progresses[i] > agent_progresses[self.actor_id]:
                decision: int = self.detector.evalueate(ego_traj, traj)
        self.observation["hazard_coeff"] = decision

    def handle_action(self, action: np.ndarray) -> None:
        self.leds[0], self.leds[3] = not self.leds[0], not self.leds[3]
        throttle: float = action[0] * self.throttle_coeff * self.observation["hazard_coeff"]
        steering: float = action[1] * self.steering_coeff
        self.running_gear.read_write_std(self.qlabs, throttle, steering, self.leds)
        self.observation["action"] = action

    def step(
        self, 
        agent_states: List[np.ndarray], 
        agent_trajs: np.ndarray,
        agent_progresses: np.ndarray
    ) -> None:
        self._get_ego_state(agent_states)
        self.handle_avoid_collide(agent_trajs, agent_progresses)
        self._handle_preprocess()
        action, _ = self.policy.execute(self.observation)
        self.handle_action(action)
