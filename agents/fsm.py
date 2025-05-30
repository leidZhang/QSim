import time
from abc import ABC, abstractmethod
from typing import List, Any, Dict

from shapely.geometry import Polygon, Point

from .interfaces import RunningGear

CRUISE_STATE_ID: int = 0
OBSERVE_STATE_ID: int = 1
STOP_STATE_ID: int = 2
SPEEDING_STATE_ID: int = 3

ZERO_PWM: float = 0
CRUISE_PWM: float = 0.08
SPEEDING_PWM: float = 0.20

class State(ABC):
    def __init__(self, id: int, description: str = "") -> None:
        self.id: int = id
        self.description: str = description

    def reset(self) -> None:
        ...

    @abstractmethod
    def act(self, *args) -> Any:
        ...

    @abstractmethod
    def transition(self, *args) -> int:
        ...

    def __str__(self) -> str:
        return f"State {self.id}: {self.description}"

class CruiseState(State):
    def __init__(self, id: int = CRUISE_STATE_ID, cruise_speed: float = CRUISE_PWM) -> None:
        super().__init__(id, "cruise")
        self.cruise_speed: float = cruise_speed
        self.has_observed: bool = False

    def reset(self) -> None:
        self.has_observed = False

    def act(self, running_gear: RunningGear, obs: Dict[str, Any]) -> Any:
        steering: float = obs['action'][1]
        obs['action'][0] = self.cruise_speed
        running_gear.read_write_std(self.cruise_speed, steering)

    def transition(
        self, 
        obs: Dict[str, Any],
        pending_area: Polygon,
        pending_flag: bool
    ) -> int:
        if obs['done']:
            # print("Transition to STOP state")
            return STOP_STATE_ID
        if not self.has_observed and pending_area.contains(Point(obs['state_info'][:2])):
            self.has_observed = True
            # print("Transition to OBSERVE state")
            return OBSERVE_STATE_ID
        return self.id
    
class ObserveState(State):
    def __init__(
        self, 
        id: int = OBSERVE_STATE_ID, 
        observe_time: float = 0.2
    ) -> None:
        super().__init__(id, "observe")
        self.observe_time: float = observe_time
        self.elapsed_time: float = 0.0
        self.started: bool = False
        self.start: float = 0.0

    def act(self, running_gear: RunningGear, obs: Dict[str, Any]) -> Any:
        steering: float = obs['action'][1]
        obs['action'][0] = ZERO_PWM
        if not self.started:
            self.started = True
            self.start: float = time.perf_counter()
        running_gear.read_write_std(ZERO_PWM, steering)

    def reset(self) -> None:
        self.elapsed_time = 0.0
        self.started = False

    def transition(
        self, 
        obs: Dict[str, Any],
        pending_area: Polygon,
        pending_flag: bool
    ) -> int:
        self.elapsed_time = time.perf_counter() - self.start
        if self.elapsed_time >= self.observe_time:
            if pending_flag:
                # print("Transition to STOP state")
                return STOP_STATE_ID
            else:
                # print("Transition to CRUISE state")
                return CRUISE_STATE_ID
        return self.id

class StopState(State):
    def __init__(self, id: int = STOP_STATE_ID) -> None:
        super().__init__(id, "stop")

    def act(self, running_gear: RunningGear, obs: Dict[str, Any]) -> Any:
        steering: float = obs['action'][1]
        obs['action'][0] = ZERO_PWM
        running_gear.read_write_std(ZERO_PWM, steering)

    def transition(
        self, 
        obs: Dict[str, Any],
        pending_area: Polygon,
        pending_flag: bool
    ) -> int:
        if not pending_flag and not obs['done']:
            # print("Transition to CRUISE state")
            return CRUISE_STATE_ID
        return self.id
    
class SpeedingState(State):
    def __init__(self, id: int = SPEEDING_STATE_ID, speeding_speed: float = SPEEDING_PWM) -> None:
        super().__init__(id, "speeding")
        self.speeding_speed: float = speeding_speed

    def act(self, running_gear: RunningGear, obs: Dict[str, Any]) -> Any:
        steering: float = obs['action'][1]
        obs['action'][0] = self.speeding_speed
        running_gear.read_write_std(self.speeding_speed, steering)

    def transition(
        self, 
        obs: Dict[str, Any],
        pending_area: Polygon,
        pending_flag: bool
    ) -> int:
        point: Point = Point(obs['state_info'][:2])
        if pending_area.contains(point):
            # print("Transition to OBSERVE state")
            return OBSERVE_STATE_ID
        return self.id
    

class FSM:
    def __init__(
        self, 
        running_gear: RunningGear,
        init_state_id: int = CRUISE_STATE_ID,
    ) -> None:
        self.running_gear: RunningGear = running_gear
        self.current_state_id: int = init_state_id
        self.states: Dict[int, State] = {
            CRUISE_STATE_ID: CruiseState(),
            OBSERVE_STATE_ID: ObserveState(),
            STOP_STATE_ID: StopState(),
            SPEEDING_STATE_ID: SpeedingState()
        }

    def reset(self) -> None:
        for state in self.states.values():
            state.reset()
        # self.current_state_id = CRUISE_STATE_ID

    def step(
        self, 
        obs: Dict[str, Any], 
        pending_area: Polygon, 
        pending_flag: bool
    ) -> None:
        current_state: State = self.states[self.current_state_id]
        current_state.act(self.running_gear, obs)
        self.current_state_id = current_state.transition(obs, pending_area, pending_flag)
            
    def set_init_state(self, state_id: int) -> None:
        if state_id not in self.states.keys():
            raise ValueError(f"Invalid state id: {state_id}")
        self.current_state_id = state_id
        print(f"FSM initial state set to {self.states[state_id]}")
