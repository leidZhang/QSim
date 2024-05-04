class InsufficientDataException(Exception): 
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class StopTrainingException(Exception): 
    def __init__(self, *args: object) -> None:
        super().__init__(*args)