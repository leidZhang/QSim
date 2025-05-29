from typing import Dict

ACTION_PROIORITY: Dict[str, int] = {
    "left": 0,
    "straight": 2,
    "right": 1
} # Traffic rule

class PriorityNode:
    def __init__(self, id: int) -> None:
        self.left: PriorityNode = None
        self.right: PriorityNode = None
        self.id: int = id
        self.action: str = None

    def set_left(self, left: 'PriorityNode') -> None:
        self.left = left

    def set_right(self, right: 'PriorityNode') -> None:
        self.right = right

    def set_action(self, action: str) -> None:
        self.action = action 

    # TODO: Modify this funtion 
    def has_higher_priority(self, other: 'PriorityNode' = None) -> bool:
        if other is None or type(other) != PriorityNode:
            raise ValueError("Invalid other node, other node should be a PriorityNode object")

        if ACTION_PROIORITY[self.action] > ACTION_PROIORITY[other.action]:
            return True

        if self.left is not None and self.left == other: # come from left
            return True
        if self.right is not None and self.right == other: # come from right
            return False

        if ACTION_PROIORITY[self.action] == ACTION_PROIORITY[other.action]:
            return self.id < other.id

        return False
    
    def __str__(self):
        return f"PriorityNode(id={self.id}, action={self.action})"



