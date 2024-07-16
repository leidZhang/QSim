from typing import List, Dict, Any, Set, Tuple
from threading import Event

import pytest

from core.control.automata import BaseState, EventDrivenDFA


# -------- Mock Classes for Testing --------
class MockCar:
    def __init__(self, init_policy: str = "RL Model") -> None:
        self.policy: str = init_policy

    def switch_to_expert(self) -> None:
        self.policy = 'PurePersuit'
        print(f"Switch to expert control!")

    def switch_to_agent(self) -> None:
        self.policy = 'RL Model'
        print(f"Switch to model control!")

    def handle_led(self) -> None:
        print("Turing light...")

    def handle_emergency_flash(self) -> None:
        print("Emergency flash...")

    def terminate(self) -> None:
        print("Terminating car...")

    def halt_car(self) -> None:
        print("Halting Car...")

    def handle_action(self) -> None:
        print(f"{self.policy} processing observation...")

    def handle_saving_data(self) -> None:
        print("Saving step data...")

    def handle_transmit_data(self) -> None:
        print("Transmiting data...")


class MockModelControl(BaseState):
    def __init__(self, car: MockCar, events: list, id: int = 0) -> None:
        transitions: Dict[int, Dict[str, Any]] = {
            1: {'trigger': events[1], 'pre_transition': car.switch_to_expert},
            2: {'trigger': events[2], 'pre_transition': car.halt_car},
            3: {'trigger': events[3], 'pre_transition': car.halt_car}
        }
        super().__init__(transitions, id)

    def handle_action(self, car: MockCar) -> None:
        car.handle_action()
        car.handle_led()


class MockPurePersuitControl(BaseState):
    def __init__(self, car: MockCar, events: list, id: int = 1) -> None:
        transitions: Dict[int, Dict[str, Any]] = {
            0: {'trigger': events[0], 'pre_transition': car.switch_to_agent},
            2: {'trigger': events[2], 'pre_transition': car.halt_car},
            3: {'trigger': events[3], 'pre_transition': car.halt_car}
        }
        super().__init__(transitions, id)

    def handle_action(self, car: MockCar) -> None:
        car.handle_action()
        car.handle_saving_data()
        car.handle_led()


class MockDataTransmission(BaseState):
    def __init__(self, car: MockCar, events: list, id: int = 2) -> None:
        transitions: Dict[int, Dict[str, Any]] = {
            0: {'trigger': events[0], 'pre_transition': car.switch_to_agent},
            # 1: {'trigger': events[1], 'pre_transition': car.switch_to_expert},
            3: {'trigger': events[3], 'pre_transition': car.halt_car}
        }
        super().__init__(transitions, id)

    def handle_action(self, car: MockCar) -> None: 
        car.handle_transmit_data()
        car.handle_emergency_flash()


class MockFinal(BaseState): 
    def __init__(self, id: int = 3) -> None:   
        transitions: Dict[int, Dict[str, Any]] = {}
        super().__init__(transitions, id)

    def handle_action(self, car: MockCar) -> Any:    
        car.terminate()


def prepare_automaton(events: List[Event], instance: Any) -> EventDrivenDFA:
    states: List[BaseState] = [
        MockModelControl(instance, events),
        MockPurePersuitControl(instance, events),
        MockDataTransmission(instance, events),
        MockFinal()
    ]
    final_states: Set[BaseState] = set([3])

    return EventDrivenDFA(states, final_states, 0)


def prepare_instance() -> Tuple[List[Event], MockCar, BaseState]:
    events: List[Event] = [Event() for _ in range(4)]
    car: MockCar = MockCar()
    state: BaseState = MockModelControl(car, events)
    return events, car, state


# -------- Test Cases for States --------
def test_state_init() -> None:
    # test the condition when trying to transit to self
    with pytest.raises(IndexError, match=f"Cannot transit to self!"):
        events: List[Event] = [Event() for _ in range(4)]
        car: MockCar = MockCar()
        state: BaseState = MockModelControl(car, events, 2)


def test_determine_next_state_1() -> None:
    # test function when no transition happens
    _, _, state = prepare_instance()
    expected_next_state: int = -1

    next_state: int = state.determine_next_state()
    assert next_state == expected_next_state


def test_determine_next_state_2() -> None:
    # test function when transition happens
    events, _, state = prepare_instance()
    expected_next_state: int = 1

    events[1].set()
    next_state: int = state.determine_next_state()
    assert next_state == expected_next_state


def test_perform_transitions_1(capsys) -> None:
    # test function when next index exists
    events, _, state = prepare_instance()
    expected_event_condition: bool = True
    expected_output_str: str = "Switch to expert control!\n"

    events[1].set()
    next_state: int = state.determine_next_state()
    state.perform_transition(next_state)
    captured = capsys.readouterr()

    assert (events[1].is_set() == expected_event_condition)
    assert captured.out == expected_output_str


def test_perform_transitions_2() -> None:
    # test function when next index does not exist
    events, _, state = prepare_instance()
    error_state_index: int = 9999

    with pytest.raises(IndexError, match=f"{error_state_index} does not exist!"):
        events[1].set()
        state.perform_transition(error_state_index)


# -------- Test Cases for EDDFA --------
def test_set_current_state_1() -> None:
    # test function when state id is valid
    events, car, _ = prepare_instance()
    fsm: EventDrivenDFA = prepare_automaton(events=events, instance=car)
    expected_current_state_id: int = 1

    fsm.set_current_state(1)
    assert fsm.current_state.id == expected_current_state_id


def test_set_current_state_2() -> None:
    # test function when state id is not valid
    events, car, _ = prepare_instance()
    fsm: EventDrivenDFA = prepare_automaton(events=events, instance=car)
    error_state_index: int = 9999

    with pytest.raises(ValueError, match=f"State does not exist!"):
        fsm.set_current_state(error_state_index)