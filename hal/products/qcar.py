"""This module contains QCar specific implementations of hal features"""
from pytransform3d import rotations as pr
import numpy as np
import time
import os

from pal.utilities.math import wrap_to_pi
from hal.utilities.estimation import EKF, KalmanFilter
from hal.utilities.control import PID, StanleyController
from hal.utilities.geometry import MobileRobotGeometry


class QCarEKF:
    """ An EKF designed to estimate the 2D position and orientation of a QCar.

    Attributes:
        kf (KalmanFilter): Kalman filter for orientation estimation.
        ekf (EKF): Extended Kalman filter for pose estimation.
        L (float): Wheelbase of the vehicle.
        x_hat (ndarray): State estimate vector [x; y; theta].
    """

    def __init__(
            self,
            x_0,
            Q_kf=np.diagflat([0.0001, 0.001]),
            R_kf=np.diagflat([.001]),
            Q_ekf=np.diagflat([0.01, 0.01, 0.01]),
            R_ekf=np.diagflat([0.01, 0.01, 0.001])
        ):
        """Initialize QCarEKF with initial state and noise covariance matrices.

        Args:
            x_0 (ndarray): Initial state vector [x, y, theta].
            Q_kf (ndarray, optional): KF process noise covariance matrix.
            R_kf (ndarray, optional): KF measurement noise covariance matrix.
            Q_ekf (ndarray, optional): EKF process noise covariance matrix.
            R_ekf (ndarray, optional): EKF measurement noise covariance matrix.
        """

        x_0 = np.squeeze(x_0)
        self.kf = KalmanFilter(
            x_0=[x_0[2], 0],
            P0=np.eye(2),
            Q=Q_kf,
            R=R_kf,
            A=np.array([[0, -1], [0, 0]]),
            B=np.array([[1], [0]]),
            C=np.array([[1, 0]])
        )

        self.ekf = EKF(
            x_0=x_0,
            P0=np.eye(3),
            Q=Q_ekf,
            R=R_ekf,
            f=self.f,
            J_f=self.J_f,
            C=np.eye(3)
        )

        self.L = 0.2
        self.x_hat = self.ekf.x_hat

    def f(self, x, u, dt):
        """Motion model for the kinematic bicycle model.

        Args:
            x (ndarray): State vector [x, y, theta].
            u (ndarray): Control input vector [v, delta].
            dt (float): Time step in seconds.

        Returns:
            ndarray: Updated state vector after applying motion model.
        """

        return x + dt * u[0] * np.array([
            [np.cos(x[2,0])],
            [np.sin(x[2,0])],
            [np.tan(u[1]) / self.L]
        ])

    def J_f(self, x, u, dt):
        """Jacobian of the motion model for the kinematic bicycle model.

        Args:
            x (ndarray): State vector [x, y, theta].
            u (ndarray): Control input vector [v, delta].
            dt (float): Time step in seconds.

        Returns:
            ndarray: Jacobian matrix of the motion model.
        """

        return np.array([
            [1, 0, -dt*u[0]*np.sin(x[2,0])],
            [0, 1, dt*u[0]*np.cos(x[2,0])],
            [0, 0, 1]
        ])

    def update(self, u=None, dt=None, y_gps=None, y_imu=None):
        """Update the EKF state estimate using GPS and IMU measurements.

        Args:
            u (ndarray, optional): Control input vector [v, delta].
            dt (float, optional): Time step in seconds.
            y_gps (ndarray, optional): GPS measurement vector [x, y, th].
            y_imu (float, optional): IMU measurement of orientation.
        """

        if dt is not None:
            if y_imu is not None:
                self.kf.predict(y_imu, dt)
                self.kf.x_hat[0,0] = wrap_to_pi(self.kf.x_hat[0,0])
            if u is not None:
                self.ekf.predict(u, dt)
                self.ekf.x_hat[2,0] = wrap_to_pi(self.ekf.x_hat[2,0])

        if y_gps is not None:
            y_gps = np.squeeze(y_gps)

            y_kf = (
                wrap_to_pi(y_gps[2] - self.kf.x_hat[0,0])
                + self.kf.x_hat[0,0]
            )
            self.kf.correct(y_kf, dt)
            self.kf.x_hat[0,0] = wrap_to_pi(self.kf.x_hat[0,0])

            y_ekf = np.array([
                [y_gps[0]],
                [y_gps[1]],
                [self.kf.x_hat[0,0]]
            ])
            z_ekf = y_ekf - self.ekf.C @ self.ekf.x_hat
            z_ekf[2] = wrap_to_pi(z_ekf[2])
            y_ekf = z_ekf + self.ekf.C @ self.ekf.x_hat
            self.ekf.correct(y_ekf, dt)
            self.ekf.x_hat[2,0] = wrap_to_pi(self.ekf.x_hat[2,0])

        else:
            y_ekf = (
                wrap_to_pi(self.kf.x_hat[0,0] - self.ekf.x_hat[2,0])
                + self.ekf.x_hat[2,0]
            )
            self.ekf.correct([None, None, y_ekf], dt)
            self.ekf.x_hat[2,0] = wrap_to_pi(self.ekf.x_hat[2,0])

        self.x_hat = self.ekf.x_hat

class QCarDriveController:
    """Implements a drive controller for a QCar that handles speed and steering

    Attributes:
        speedController (PID): PI controller for speed control.
        steeringController (StanleyController): Nonlinear Stanley controller
            for steering control.
    """

    def __init__(self, waypoints, cyclic):
        """Initialize QCarDriveController

        Args:
            waypoints (list): List of waypoints for the controller to follow.
            cyclic (bool): Indicates if the waypoint path is cyclic or not.
        """

        self.speedController = PID(
            Kp=0.1,
            Ki=1,
            Kd=0,
            uLimits=(-0.3, 0.3)
        )

        self.steeringController = StanleyController(
            waypoints=waypoints,
            k=1,
            cyclic=cyclic
        )
        self.steeringController.maxSteeringAngle = np.pi/6

    def reset(self):
        """Resets the internal state of the speed and steering controllers."""

        self.speedController.reset()
        self.steeringController.wpi = 0
        self.steeringController.pathComplete = False

    def updatePath(self, waypoints, cyclic):
        """Updates the waypoint path for the steering controller.

        Args:
            waypoints (list): List of new waypoints to follow
            cyclic (bool): Indicates if the updated path is cyclic or not.
        """

        self.steeringController.updatePath(waypoints, cyclic)

    def update(self, p, th, v, v_ref, dt):
        """Updates the drive controller with the current state of the QCar.

        Args:
            p (ndarray): Position vector [x, y].
            th (float): Orientation angle in radians.
            v (float): Current speed of the QCar.
            v_ref (float): Reference speed for the QCar.
            dt (float): Time step in seconds.

        Returns:
            float: Speed control input.
            float: Steering control input.
        """

        if not self.steeringController.pathComplete:
            delta = self.steeringController.update(p, th, v)
        else:
            delta = 0
            v_ref = 0

        u = self.speedController.update(v_ref, v, dt)

        return u, delta

class QCarGeometry(MobileRobotGeometry):
    """QCarGeometry class for defining QCar-specific frames of reference.

    This class inherits from the MobileRobotGeometry class and adds frames
    specific to the QCar, such as the front and rear axles, CSI sensors,
    IMU, Realsense, and RPLidar.
    """

    def __init__(self):
        """Initialize the QCarGeometry class with QCar-specific frames."""

        super().__init__()

        self.defaultFrame = 'body'

        # Body reference frame center is located at,
        # X: halfway between front and rear axles
        # Y: halway between left and right wheels
        # Z: 0.0103 m above the ground (chassis bottom)
        # the following frames are all defined w.r.t this center
        self.add_frame(
            name='CG',
            p=[0.0248, -0.0074, 0.0606],
            R=np.eye(3)
        )
        self.add_frame(
            name='front_axle',
            p=[0.1300, 0, 0.0207],
            R=np.eye(3)
        )
        self.add_frame(
            name='rear_axle',
            p=[-0.1300, 0, 0.0207],
            R=np.eye(3)
        )
        self.add_frame(
            name='csi_front',
            p=[0.1930, 0, 0.0850],
            R=pr.active_matrix_from_extrinsic_euler_xyz(
                [-np.pi/2, 0, -np.pi/2]
            )
        )
        self.add_frame(
            name='csi_left',
            p=[0.0140, 0.0438, 0.0850],
            R=pr.active_matrix_from_extrinsic_euler_xyz([-np.pi/2, 0, 0])
        )
        self.add_frame(
            name='csi_rear',
            p=[-0.1650, 0, 0.0850],
            R=pr.active_matrix_from_extrinsic_euler_xyz([-np.pi/2, 0, np.pi/2])
        )
        self.add_frame(
            name='csi_right',
            p=[0.0140, -0.0674, 0.0850],
            R=pr.active_matrix_from_extrinsic_euler_xyz([-np.pi/2, 0, np.pi])
        )
        self.add_frame(
            name='imu',
            p=[0.1278, 0.0223, 0.0792],
            R=np.eye(3)
        )
        self.add_frame(
            name='realsense',
            p=[0.0822, 0.0003, 0.1479],
            R=pr.active_matrix_from_extrinsic_euler_xyz(
                [-np.pi/2, 0, -np.pi/2]
            )
        )
        self.add_frame(
            name='rplidar',
            p=[-0.0108, 0, 0.1696],
            R=np.eye(3)
        )

        self.defaultFrame = 'world'