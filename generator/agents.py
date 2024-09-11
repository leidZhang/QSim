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
        self.observation: Dict[str, np.ndarray] = {"action": np.zeros(2)}

    def _get_ego_state(self, agent_states: List[np.ndarray]) -> None:
        ego_state: np.ndarray = agent_states[self.actor_id]
        hazard_states: List[np.ndarray] = [
            state for i, state in enumerate(agent_states) if i != self.actor_id and i != 0
        ]

        self.state = ego_state
        self.observation["hazards"] = hazard_states

    def reset(self, wayponts: np.ndarray, agent_states: List[np.ndarray]) -> None:
        self._get_ego_state(agent_states)
        self.observation = self.preporcessor.setup(
            ego_state=self.state,
            observation=self.observation,
            waypoints=wayponts,
        )
        self.observation["global_waypoints"] = self.preporcessor.global_waypoints

    def _handle_preprocess(self) -> None:
        self.observation = self.preporcessor.execute(self.state, self.observation)
        self.observation["global_waypoints"] = self.preporcessor.global_waypoints

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

    def step(self, agent_states: List[np.ndarray]) -> None:
        self.__get_image_data()
        self._get_ego_state(agent_states)
        self._handle_preprocess()
        action, _ = self.policy.execute(self.observation)
        self.handle_action(action)


class HazardAgent(CarAgent):
    def __init__(self, actor_id: int, qlabs: QuanserInteractiveLabs) -> None:
        super().__init__(actor_id)
        self.qlabs: QuanserInteractiveLabs = qlabs
        self.detector: HazardDetector = HazardDetector()
        self.running_gear: VirtualRuningGear = VirtualRuningGear(QCAR_ACTOR_ID, actor_id)

    def _detect_hazard(self) -> None:
        ego_state: np.ndarray = self.observation["state"]
        hazard_decision: int = self.detector.get_hazard_decision(
            ego_state, self.observation["hazards"]
        )
        self.observation["hazard_decision"] = hazard_decision
        print(f"Agent {self.actor_id} has Hazard decision: {hazard_decision}")

    def halt_car(self, steering: float = 0, halt_time: float = 0.1) -> None:
        self.running_gear.read_write_std(self.qlabs, throttle=0.0, steering=steering)
        time.sleep(halt_time)

    def handle_action(self, action: np.ndarray) -> None:
        self.leds[0], self.leds[3] = not self.leds[0], not self.leds[3]
        throttle: float = action[0] * self.throttle_coeff * self.observation["hazard_decision"]
        steering: float = action[1] * self.steering_coeff
        self.running_gear.read_write_std(self.qlabs, throttle, steering, self.leds)
        self.observation["action"] = action

    def step(self, agent_states: List[np.ndarray]) -> None:
        print(f"Agent {self.actor_id} is running...")
        self._get_ego_state(agent_states)
        self._detect_hazard()
        self._handle_preprocess()
        action, _ = self.policy.execute(self.observation)
        self.handle_action(action)
