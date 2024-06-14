class TachFilter:
    def __init__(self, threshold: float = 0.4) -> None:
        self.last_signal: float = 0.0
        self.threshold: float = threshold

    def reset(self) -> None:
        self.last_signal = 0.0

    def _is_not_noise(self, signal: float) -> bool:
        return abs(signal - self.last_signal) <= self.threshold

    def __call__(self, signal: float) -> float:
        if self._is_not_noise(signal):
            self.last_signal = signal
        return self.last_signal
