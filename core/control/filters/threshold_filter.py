from collections import deque


class ThresholdFilter:
    """
    ThresholdFilter class is a simple filter for filtering the signal based on the threshold.
    The filter can be used in two modes: low pass mode or high pass mode.

    Attributes:
    - use_low_pass: bool, default=True
    - threshold: float, default=0.4
    - last_signal: float, default=0.0
    """

    def __init__(self, use_low_pass: bool = True, threshold: float = 0.4) -> None:
        """
        ThresholdFilter class constructor method, initialize the filter with the given threshold.

        Parameters:
        - use_low_pass: bool, default=True
        - threshold: float, default=0.4
        """
        # base attributes
        self.last_signal: float = 0.0
        self.threshold: float = threshold
        # low pass mode or high pass mode
        if use_low_pass:
            self.will_consider = self._low_pass
        else:
            self.will_consider = self._high_pass

    def _low_pass(self, signal: float) -> bool:
        """
        Low pass mode for the filter, return True if the signal is close to the last signal.

        Parameters:
        - signal: float: The current signal to be filtered

        Returns:
        - bool: True if the signal is close to the last signal, otherwise False
        """
        return abs(signal - self.last_signal) <= self.threshold

    def _high_pass(self, signal: float) -> bool:
        """
        High pass mode for the filter, return True if the signal is far from the last signal.

        Parameters:
        - signal: float: The current signal to be filtered

        Returns:
        - bool: True if the signal is far from the last signal, otherwise False
        """
        return abs(signal - self.last_signal) >= self.threshold

    def __call__(self, signal: float) -> float:
        """
        The main method for the filter, filter the signal based on the threshold.

        Parameters:
        - signal: float: The current signal to be filtered

        Returns:
        - float: The filtered signal
        """
        if self.will_consider(signal):
            self.last_signal = signal
        return self.last_signal

    def reset(self) -> None:
        """
        Reset the filter to the initial state.

        Returns:
        - None
        """
        self.last_signal = 0.0


class VariableThresholdFilter(ThresholdFilter):
    """
    The VariableThresholdFilter class is a filter that uses a variable threshold based on the signal history.

    Attributes:
    - variable_threshold: float, default=0.4
    - reduce_factor: float, default=0.3
    - buffer_size: int, default=10
    - buffer: deque
    - accumulator: float, default=0.0
    - last_avg: float, default=0.0
    """

    def __init__(
            self,
            use_low_pass: bool = True,
            threshold: float = 0.4,
            reduce_factor: float = 0.3,
            buffer_size: int = 10
        ) -> None:
        """
        The VariableThresholdFilter class constructor, initialize the filter with the given parameters.

        Parameters:
        - use_low_pass: bool, default=True
        - threshold: float, default=0.4
        - reduce_factor: float, default=0.3
        - buffer_size: int, default=10
        """
        super().__init__(use_low_pass, threshold)
        # variable signal for stable or changing state
        self.variable_threshold: float = self.threshold
        self.reduce_factor: float = reduce_factor
        # low pass mode or high pass mode
        if use_low_pass:
            self.is_not_noise = self._is_lower
        else:
            self.is_not_noise = self._is_higher
        # buffer for valid history signals
        self.buffer_size: int = buffer_size
        self.buffer: deque = deque(maxlen=self.buffer_size)
        self.accumulator: float = 0.0
        # last avg for compare
        self.last_avg: float = 0.0

    def _is_lower(self, signal: float) -> bool:
        """
        Check if the signal is close to the last signal.

        Parameters:
        - signal: float: The current signal to be filtered

        Returns:
        - bool: True if the signal is close to the last signal, otherwise False
        """
        return abs(signal - self.last_signal) <= self.variable_threshold

    def _is_higher(self, signal: float) -> bool:
        """
        Check if the signal is far from the last signal.

        Parameters:
        - signal: float: The current signal to be filtered

        Returns:
        - bool: True if the signal is far from the last signal, otherwise False
        """
        return abs(signal - self.last_signal) >= self.variable_threshold

    def _add_to_buffer(self, signal: float) -> None:
        """
        Add the signal to the buffer and update the accumulator.

        Parameters:
        - signal: float: The current signal to be filtered

        Returns:
        - None
        """
        print(f"Incomming: {signal}, last: {self.last_signal} Will consider: {self.will_consider(signal)}")
        # use last signal if is noise
        if not self.will_consider(signal) and len(self.buffer) > 0:
            signal = self.buffer[0]
        # push signal to the buffer
        if len(self.buffer) == self.buffer.maxlen:
            self.accumulator -= self.buffer.pop()
        self.buffer.appendleft(signal)
        # update the accumulator
        self.accumulator += signal

    def _is_stable(self) -> bool:
        """
        Check if the signal is stable based on the history.

        Returns:
        - bool: True if the signal is stable, otherwise False
        """
        # calculate the current average value
        avg: float = self.accumulator / len(self.buffer)
        # compare the current average and the last average
        flag: bool = abs(avg - self.last_avg) <= 0.1
        # update the last average
        self.last_avg = avg
        return flag

    def __call__(self, signal: float) -> float:
        """
        The main method for the filter, filter the signal based on the variable threshold.

        Parameters:
        - signal: float: The current signal to be filtered

        Returns:
        - float: The filtered signal
        """
        # add signal to the buffer
        self._add_to_buffer(signal)
        # update variable threshold
        if self._is_stable():
            self.variable_threshold = self.threshold * self.reduce_factor
        else:
            self.variable_threshold = self.threshold
        # final decision based on the variable threshold
        if self.is_not_noise(signal):
            self.last_signal = signal
        return self.last_signal

    def reset(self) -> None:
        """
        Reset the filter to the initial state.

        Returns:
        - None
        """
        super().reset()
        # reset to initial state
        self.buffer = deque(maxlen=self.buffer_size)
        self.accumulator: float = 0.0
        self.last_avg: float = 0.0
