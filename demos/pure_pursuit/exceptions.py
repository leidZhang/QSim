class ReachGoalException(Exception):
    def __init__(self, message: object) -> None:
        super().__init__(message)