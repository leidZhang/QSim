from collections import deque

import numpy as np

from pal.utilities.math import wrap_to_pi


class ExtendedKalmanFilter: # quanser's code
    def __init__(self, x0, P0, Q, R):
        # Nomenclature:
        # - x0: initial estimate
        # - P0: initial covariance matrix estimate
        # - Q: process noise covariance matrix
        # - R: observation noise covariance matrix
        # - xHat: state estimate
        # - P: state covariance matrix
        # - L: wheel base of the QCar
        # - C: output matrix
        self.L = 0.257

        self.I = np.eye(3)
        self.xHat = x0
        self.P = P0
        self.Q = Q
        self.R = R

        self.C = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])

    def f(self, X, u, dt):
        # Kinematic Bicycle Model:
        # - X = [x, y, theta]
        # - u[0] = v (speed in [m/s])
        # - u[1] = delta (steering Angle in [rad])
        # - dt: change in time since last update
        return X + dt * u[0] * np.array([
            [np.cos(X[2,0])],
            [np.sin(X[2,0])],
            [np.tan(u[1]) / self.L]
        ])

    def Jf(self, X, u, dt):
        # Jacobian for the kinematic bicycle model (see self.f)
        return np.array([
                [1, 0, -dt*u[0]*np.sin(X[2,0])],
                [0, 1, dt*u[0]*np.cos(X[2,0])],
                [0, 0, 1]
        ])

    def prediction(self, dt, u):
        # Update Covariance Estimate
        F = self.Jf(self.xHat, u, dt)
        self.P = F@self.P@np.transpose(F) + self.Q

        # Update State Estimate
        self.xHat = self.f(self.xHat, u, dt)
        # Wrap th to be in the range of +/- pi
        self.xHat[2] = wrap_to_pi(self.xHat[2])

    def correction(self, y):
        # Precompute terms that will be used multiple times
        H = self.C
        P_times_HTransposed = self.P @ np.transpose(H)

        S = H @ P_times_HTransposed + self.R
        K = P_times_HTransposed @ np.linalg.inv(S)

        # Wrap z for th to be in the range of +/- pi
        z = (y - H@self.xHat)
        if len(y) > 1:
            z[2] = wrap_to_pi(z[2])
        else:
            z = wrap_to_pi(z)

        self.xHat += K @ z
        # Wrap th to be in the range of +/- pi
        self.xHat[2] = wrap_to_pi(self.xHat[2])

        self.P = (self.I - K@H) @ self.P


class ThresholdFilter:
    def __init__(self, use_low_pass: bool = True, threshold: float = 0.4) -> None:
        # base attributes
        self.last_signal: float = 0.0
        self.threshold: float = threshold
        # low pass mode or high pass mode
        if use_low_pass:
            self.will_consider = self._low_pass
        else:
            self.will_consider = self._high_pass

    def _low_pass(self, signal: float) -> bool:
        return abs(signal - self.last_signal) <= self.threshold

    def _high_pass(self, signal: float) -> bool:
        return abs(signal - self.last_signal) >= self.threshold

    def __call__(self, signal: float) -> float:
        if self.will_consider(signal):
            self.last_signal = signal
        return self.last_signal

    def reset(self) -> None:
        self.last_signal = 0.0


class VariableThresholdFilter(ThresholdFilter):
    def __init__(
            self,
            use_low_pass: bool = True,
            threshold: float = 0.4,
            reduce_factor: float = 0.3,
            buffer_size: int = 10
        ) -> None:
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
        return abs(signal - self.last_signal) <= self.variable_threshold

    def _is_higher(self, signal: float) -> bool:
        return abs(signal - self.last_signal) >= self.variable_threshold

    def _add_to_buffer(self, signal: float) -> None:
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
        # calculate the current average value
        avg: float = self.accumulator / len(self.buffer)
        # compare the current average and the last average
        flag: bool = abs(avg - self.last_avg) <= 0.1
        # update the last average
        self.last_avg = avg
        return flag

    def __call__(self, signal: float) -> float:
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
        super().reset()
        # reset to initial state
        self.buffer = deque(maxlen=self.buffer_size)
        self.accumulator: float = 0.0
        self.last_avg: float = 0.0
