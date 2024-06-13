from queue import Queue


class TachFilter:
    def __init__(self, thresh_hold: float = 0.4) -> None:
        self.last_signal: float = 0.0
        self.thresh_hold: float = thresh_hold

    def is_not_noise(self, signal: float) -> bool:
        return abs(signal - self.last_signal) <= self.thresh_hold
    
    def __call__(self, signal: float) -> float:
        if self.is_not_noise(signal):
            # self.add_to_buffer(signal)
            self.last_signal = signal
        return self.last_signal
