from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseState(ABC):
    """
    BaseState is an abstract class that represents a state in a finite automaton.
    The state is defined by an id, a list of transitions, and a method to handle 
    actions. The state is designed to be used in a deterministic finite automaton (DFA)
    in such a way that the transitions are triggered by events and cannot transit to itself.

    Attributes:
    - transitions: Dict[int, Dict[str, Any]]: List of transitions from the state, 
    transition format: id: {'trigger': Event(), 'pre_transition': BaseState()}
    - id: int: Unique identifier of the state
    """
    
    def __init__(self, transitions: Dict[int, Dict[str, Any]], id: int) -> None:
        """
        Initialize the state with the given transitions and id.

        Parameters:
        - transitions: List[Dict[int, Any]]: List of transitions from the state
        - id: int: Unique identifier of the state

        Returns:
        - None

        Raises:
        - IndexError: If the state id is in the transitions
        """
        self.transitions: Dict[int, Dict[str, Any]] = transitions
        if id in self.transitions.keys():
            raise IndexError("Cannot transit to self!")
        self.id: int = id

    def determine_next_state(self) -> int:
        """
        Determine the next state based on the triggers of the transitions.

        Returns:
        - int: The id of the next state, -1 if no transition is triggered
        """
        for dest, transition in self.transitions.items():
            if transition['trigger'].is_set():
                return dest
        return -1

    def perform_transition(self, next_state: int) -> None:
        """
        Perform the transition to the next state.

        Parameters:
        - next_state: int: The id of the next state

        Returns:
        - None

        Raises:
        - IndexError: If the next state is not in the transitions
        """
        if next_state not in self.transitions.keys():
            raise IndexError(f"{next_state} does not exist!")
        self.transitions[next_state]['pre_transition']()
        # self.transitions[next_state]['trigger'].clear()

    @abstractmethod
    def handle_action(self, *args) -> Any:
        """
        Handle the action of the state. This method should be implemented by the subclass.

        Parameters:
        - args: Any: Any number of arguments to be passed to the state actions, it is expected
        to pass at least one actor instance to the state action

        Returns:
        - Any: The result of the state action
        """
        ...
