import os
import time
import math
from typing import List, Dict

import numpy as np

try:
    from quanser.common import Timeout
except:
    from quanser.communications import Timeout
from pal.utilities.stream import BasicStream
from pal.products.qcar import QCarGPS, IS_PHYSICAL_QCAR
from pal.utilities.vision import Camera2D
from pal.utilities.vision import Camera3D

from .constants import CSI_CAMERA_SETTING
from .constants import RGBD_CAMERA_SETTING


class CSICamera: # wrapper class, implement more functions if needed
    """
    CSI Camera class for reading images from the camera. It is a wrapper class for
    Camera2D class.

    Attributes:
    - id: int: camera id
    - camera: Camera2D: camera object
    """

    def __init__(self, id: int = 3, camera_setting: dict = CSI_CAMERA_SETTING) -> None:
        """
        Initialize the camera object with the given camera id and camera settings.

        Parameters:
        - id: int: camera id
        - camera_setting: dict: camera setting dictionary
        """
        self.id: int = id
        if IS_PHYSICAL_QCAR:
            camera_id: str = str(id)
        else:
            camera_id: str = str(id) + "@tcpip://localhost:" + str(18961+id)

        self.camera: Camera2D = Camera2D(
            cameraId=camera_id,
            frameWidth=camera_setting['frame_width'],
            frameHeight=camera_setting['frame_height'],
            frameRate=camera_setting['frame_rate'],
            focalLength=camera_setting['focal_length'],
            principlePoint=camera_setting['principle_point'],
            position=camera_setting['position'],
            orientation=camera_setting['orientation']
        )

    def terminate(self) -> None:
        """
        Terminate the camera object.

        Returns:
        - None
        """
        self.camera.terminate()

    def read_image(self) -> np.ndarray: # image without any modification
        """
        Read the image from the camera. This function is a non-blocking function,
        if the camera is not ready, it will return None.

        Returns:
        - np.ndarray: image data
        """
        if self.camera.read():
            return self.camera.imageData
        return None

    def await_image(self) -> np.ndarray:
        """
        Read the image from the camera. This function is a blocking function,
        it will wait until the camera is ready.

        Returns:
        - np.ndarray: image data
        """
        while not self.camera.read():
            pass
        return self.camera.imageData


class RGBDCamera:
    """
    RGBD Camera class for reading images from the camera. It is a wrapper class
    for Camera3D class.

    Attributes:
    - camera: Camera3D: camera object
    """

    def __init__(self, camera_setting: dict = RGBD_CAMERA_SETTING) -> None:
        """
        Initialize the camera object with the given camera settings.

        Parameters:
        - camera_setting: dict: camera setting dictionary
        """
        if IS_PHYSICAL_QCAR:
            device_id: str = str(camera_setting['device_id'][0])
        else:
            device_id: str = camera_setting['device_id']

        self.camera: Camera3D = Camera3D(
            mode=camera_setting['mode'],
            frameWidthRGB=camera_setting['frame_width_rgb'],
            frameHeightRGB=camera_setting['frame_height_rgb'],
            frameRateRGB=camera_setting['frame_rate_rgb'],
            frameWidthDepth=camera_setting['frame_width_depth'],
            frameHeightDepth=camera_setting['frame_height_depth'],
			frameRateDepth=camera_setting['frame_rate_depth'],
            deviceId=device_id
        )

    def terminate(self) -> None:
        """
        Terminate the camera object.

        Returns:
        - None
        """
        self.camera.terminate()

    def read_rgb_image(self) -> np.ndarray:
        """
        Read the RGB image from the camera. This function is a non-blocking function,
        if the camera is not ready, it will return None.

        Returns:
        - np.ndarray: RGB image data
        """
        # cv2.imshow('RGBD Image', self.camera.imageBufferRGB)
        frame = self.camera.streamRGB.get_frame()
        if frame is not None:
            frame.get_data(self.camera.imageBufferRGB)
            frame.release()
            return self.camera.imageBufferRGB
        return None

    def await_rgb_image(self) -> np.ndarray:
        """
        Await the RGB image from the camera. This function is a blocking function,
        it will wait until the camera is ready.

        Returns:
        - np.ndarray: RGB image data
        """
        if self.camera.read_RGB() != -1:
            return self.camera.imageBufferRGB

    def read_depth_image(self, data_mode='PX') -> np.ndarray:
        """
        Read the depth image from the camera. This function is a non-blocking function,
        if the camera is not ready, it will return None.

        Parameters:
        - data_mode: str: data mode (PX or M)

        Returns:
        - np.ndarray: depth image data
        """
        frame = self.camera.streamDepth.get_frame()
        if frame is None:
            return None

        if data_mode == 'PX':
            frame.get_data(self.camera.imageBufferDepthPX)
            frame.release()
            return self.camera.imageBufferDepthPX
        elif data_mode == 'M':
            frame.get_meters(self.camera.imageBufferDepthM)
            frame.release()
            return self.camera.imageBufferDepthM
        else:
            raise ValueError("Invalid data mode")

    def await_depth_image(self, data_mode='PX') -> np.ndarray:
        """
        Await the depth image from the camera. This function is a blocking function,
        it will wait until the camera is ready.

        Parameters:
        - data_mode: str: data mode (PX or M)

        Returns:
        - np.ndarray: depth image data
        """
        if self.camera.read_depth(data_mode) != -1:
            if data_mode == 'PX':
                # cv2.imshow('RGBD PX', self.camera.imageBufferDepthPX)
                return self.camera.imageBufferDepthPX
            elif data_mode == 'M':
                # cv2.imshow('RGBD M', self.camera.imageBufferDepthM)
                return self.camera.imageBufferDepthM
            else:
                raise ValueError("Invalid data mode")


# TODO: Merge with the code on the QCar
class LidarSLAM(QCarGPS):
    GPS_URI: str = "tcpip://localhost:18967"
    LIDAR_URI: str = "tcpip://localhost:18968"
    CONNECTION_TIMEOUT: int = 5  # seconds

    def __init__(self, initialPose: List[float] = [0, 0, 0], recalibrate: bool = False) -> None:
        print(f"Initializing LidarSLAM...")
        if IS_PHYSICAL_QCAR:
            self._init_lidar_to_gps(initialPose, recalibrate)

        self._timeout = Timeout(seconds=0, nanoseconds=1)

        # Setup GPS client and connect to GPS server
        self.position = np.zeros((3))
        self.orientation = np.zeros((3))

        self._gps_data = np.zeros((6), dtype=np.float32)
        gps_buffer_size: int = (self._gps_data.size * self._gps_data.itemsize)
        self._gps_client = self.__setup_client(self.GPS_URI, self._gps_data, gps_buffer_size)

        # Setup Lidar data client and connect to Lidar data server
        self.scanTime = 0
        self.angles = np.zeros(384)
        self.distances = np.zeros(384)

        self._lidar_data = np.zeros(384*2 + 1, dtype=np.float64)
        lidar_buffer_size: int = 8*(384*2 + 1)
        self._lidar_client = self.__setup_client(self.LIDAR_URI, self._lidar_data, lidar_buffer_size)

        self.enableFiltering = True
        self.angularResolution = 1*np.pi/180
        self._phi = np.linspace(0, 2*np.pi, np.int_(np.round(2*np.pi/self.angularResolution)))
        print("LidarSLAM initialized")

    def __setup_client(self, uri, data_buffer, buffer_size):
        client = BasicStream(
            uri=uri,
            agent='C',
            receiveBuffer=data_buffer,
            sendBufferSize=1,
            recvBufferSize=buffer_size,
            nonBlocking=True
        )
        t0 = time.time()
        while not client.connected:
            if time.time() - t0 > self.CONNECTION_TIMEOUT:
                print(f"Couldn't Connect to Server at {uri}")
                return None
            client.checkConnection()
        return client

    def terminate(self):
        """ Terminates the GPS client. """
        self._gps_client.terminate()
        self._lidar_client.terminate()
        if IS_PHYSICAL_QCAR:
            self._stop_lidar_to_gps()

    def _stop_lidar_to_gps(self):
        # Quietly stop qcarLidarToGPS if it is already running:
        # the -q flag kills the executable
        # the -Q flag kills quietly (no errors thrown if its not running)
        os.system(
            'sudo quarc_run -t tcpip://localhost:17000 -q -Q'
            + ' qcarLidarToGPS.rt-linux_nvidia'
        )

    def _init_lidar_to_gps(self, initialPose: List[float], recalibrate: bool) -> None:
        self.__initialPose: List[float] = initialPose
        self._stop_lidar_to_gps()

        if recalibrate:
            self.calibrate()
            # wait period to complete calibration completely
            time.sleep(8)
        time.sleep(8)
        print("Initializing GPS Server...")
        self._emulate_gps()
        time.sleep(4)
        print('GPS Server started.')

    def calibrate(self) -> None:
        print('Calibrating QCar at position ', self.__initialPose[0:2],
            ' (m) and heading ', self.__initialPose[2], ' (rad).')

        captureScanfile = os.path.join(
            '/home/nvidia/Documents/Quanser/libraries/',
            'resources/applications/QCarScanMatching/'
                + 'qcarCaptureScan.rt-linux_nvidia'
        )

        os.system(
            'sudo quarc_run -t tcpip://localhost:17000 '
            + captureScanfile + ' -d ' + os.getcwd()
        )

        print('Calibration complete.')

    def _emulate_gps(self) -> None:
        # setup the path to the qcarLidarToGPS file
        lidarToGPSfile = os.path.join(
            '/home/nvidia/Documents/Quanser/libraries/',
            'resources/applications/QCarScanMatching/'
                + 'qcarLidarToGPS.rt-linux_nvidia'
        )
        os.system(
            'sudo quarc_run -t tcpip://localhost:17000 '
            + lidarToGPSfile + ' -d ' + os.getcwd()
            + ' -pose_0 ' + str(self.__initialPose[0])
            + ',' + str(self.__initialPose[1])
            + ',' + str(self.__initialPose[2])
        )

VirtualRGBDCamera = RGBDCamera
VirtualCSICamera = CSICamera

# TODO: Refactor the class
# class VirtualGPS:
#     def __init__(self, initial_pose: List[float] = [0, 0, 0]) -> None:
#         self.gps: QCarGPS = LidarSLAM(initial_pose)
#         self.speed_vector = None

#     def terminate(self) -> None:
#         self.gps.terminate()
#         # plot_line_chart(self.speed_history[1:], 'time', 'speed', 'speed chart')

#     def get_gps_state(self) -> tuple:
#         position_x = self.gps.position[0]
#         position_y = self.gps.position[1]
#         orientation = self.gps.orientation[2]

#         return position_x, position_y, orientation

#     def calcualte_speed_vector(self, current_state, delta_t) -> tuple:
#         delta_x_sq = math.pow((current_state[0] - self.last_state[0]), 2)
#         delta_y_sq = math.pow((current_state[1] - self.last_state[1]), 2)

#         linear_speed = math.pow((delta_x_sq + delta_y_sq), 0.5) / delta_t
#         angular_speed = (current_state[2] - self.last_state[2]) / delta_t

#         return linear_speed, angular_speed

#     def setup(self) -> None:
#         # create or overwrite the log
#         open("output/gps_log.txt", "w")
#         # init states
#         self.time_stamp = time.time()
#         self.gps.readGPS() # read gps info
#         self.last_state = self.get_gps_state()

#     def read_gps_state(self) -> None:
#         # read current position
#         self.gps.readGPS()
#         current_time = time.time()
#         self.current_state = self.get_gps_state()
#         # calculate absolute speed
#         if self.current_state != self.last_state or current_time - self.time_stamp >= 0.25:
#             delta_t = current_time - self.time_stamp
#             self.speed_vector = self.calcualte_speed_vector(self.current_state, delta_t)
#             # self.speed_history.append(speed_vecotr[0])

#             # os.system("cls")
#             # print(f"delta_t: {delta_t:.4f}s")
#             # print(f"last_x: {self.last_state[0]:.2f}, last_y: {self.last_state[1]:.2f},  last_orientation: {((180 / np.pi) * self.last_state[2]):.2f}°")
#             # print(f"x: {self.current_state[0]:.2f}, y: {self.current_state[1]:.2f},  orientation: {((180 / np.pi) * self.current_state[2]):.2f}°")
#             # print(f"speed: {speed_vecotr[0]:.4f} m/s, angular speed: {speed_vecotr[1]:.4f} rad/s")

#             # update time stamp
#             self.time_stamp = current_time
#             # update position
#             self.last_state = self.current_state
# class VirtualLidar:
#     def __init__(self) -> None:
#         pass # will start implementation after the error -15 fixed
