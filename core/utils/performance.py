import time

def skip() -> None:
    pass

def mock_delay(start: float, delay: float) -> None:
    end: float = time.time() - start
    time.sleep(max(0, delay - end))

def skip_delay(start: float, delay: float) -> None:
    pass