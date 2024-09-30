import time
from typing import List, Any, Set

from .states import BaseState


class EventDrivenDFA:
    """
    This class represents an event-driven deterministic finite automaton (DFA).
    The DFA is defined by a list of states, a set of final states, and an initial state.
    The DFA can be executed by calling the execute method with the appropriate arguments.
    The state transitions will be determined by the current state and the input arguments.

    Attributes:
    - states: List[BaseState]: List of states in the DFA
    - final_states: Set[BaseState]: Set of final states in the DFA
    - current_state: int: The current state of the DFA
    """

    def __init__(
        self,
        states: List[BaseState],
        final_states: Set[int],
        init_state: int = 0
    ) -> None:
        """
        Initialize the DFA with the given states, final states, and initial state.

        parameters:
        - states: List[BaseState]: List of states in the DFA
        - final_states: Set[BaseState]: Set of final states in the DFA
        - init_state: BaseState: Index of the initial state in the states
        """
        self.states: List[BaseState] = states
        self.final_states: Set[int] = final_states
        self.current_state: BaseState = self.states[init_state]

    def execute(self, *args: Any) -> None:
        """
        Execute the DFA with the given input arguments, the state transitions will 
        be determined by the current state. If the current state is a final state, 
        the state will not be able to change.

        Parameters:
        - args: Any: Any number of arguments to be passed to the state actions

        Returns: 
        - None
        """
        # check final state
        if self.current_state.id in self.final_states:
            print(f"Currently in final state {self.current_state.id}")
            time.sleep(1)
            return

        # check state transition
        next_state: int = self.current_state.determine_next_state()
        if next_state != -1:
            print(f"Transition from {self.current_state.id} to {next_state}")
            self.current_state.perform_transition(next_state) # conditional transition
            self.current_state = self.states[next_state] # assign new state
        # execute state actions
        self.current_state.handle_action(*args)

    def get_current_state(self) -> int:
        """
        This method returns the id of the current state of the DFA.

        Returns:
        - int: The id of the current state
        """
        return self.current_state.id

    def set_current_state(self, state_id: int) -> None:
        """
        This method sets the current state of the DFA to the state with the given id.

        Parameters:
        - state_id: int: The id of the state to set as the current state

        Returns:
        - None

        Raises:
        - ValueError: If the state_id is not a valid state index
        """
        if state_id < 0 or state_id > len(self.states) - 1:
            raise ValueError("State does not exist!")
        self.current_state = self.states[state_id]
