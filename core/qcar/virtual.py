import struct

import numpy as np

from qvl.actor import QLabsActor
from qvl.qlabs import QuanserInteractiveLabs
from qvl.qlabs import CommModularContainer


class VirtualOptitrack:
    """
    VirtualOptitrack is a class that mocks the Optitrack system. It is used to get the position of 
    an actor in the Quanser Interactive Labs environment.

    Attributes:
    - dt: float: The time gap between each step
    - class_id: int: The class id of the actor
    - actor_number: int: The actor number
    - state: np.array: The state of the actor
    - last_state: np.array: The last state of the actor
    """

    def __init__(self, class_id: int, actor_number: int, dt: float = 0.05) -> None:
        """
        The constructor of the VirtualOptitrack class. Initialize the VirtualOptitrack object with the given parameters.

        Parameters:
        - class_id: int: The class id of the actor
        - actor_number: int: The actor number
        - dt: float: The time gap between each step
        """
        self.dt = dt
        self.class_id: int = class_id
        self.actor_number: int = actor_number
        self.state: np.array = np.zeros(6)
        self.last_state: np.array = np.zeros(6)

    def get_position(self, qlabs: QuanserInteractiveLabs) -> np.ndarray:
        """
        Get the position of the actor in the Quanser Interactive Labs environment.

        Parameters:
        - qlabs: QuanserInteractiveLabs: The Quanser Interactive Labs object

        Returns:
        - np.ndarray: The state of the actor
        """
        c: CommModularContainer = CommModularContainer()
        c.classID = self.class_id
        c.actorNumber = self.actor_number
        c.actorFunction = QLabsActor.FCN_REQUEST_WORLD_TRANSFORM
        c.payload = bytearray()
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        if not qlabs.send_container(c):
            raise Exception("Failed to send container")
        c = qlabs.wait_for_container(self.class_id, self.actor_number, QLabsActor.FCN_RESPONSE_WORLD_TRANSFORM)
        x, y, _, _, _, yaw, _, _, _, = struct.unpack(">fffffffff", c.payload[0:36])
        self.state = np.array([x, y, yaw, 0, 0, 0])

    def cal_motion(self) -> None:
        """
        Calculate the motion of the actor based on the current and last state.

        Returns:
        - None
        """
        x: float = self.state[0]
        y: float = self.state[1]
        yaw: float = self.state[2]
        # print(f'yaw: {yaw}')

        # calc velocity
        vx = (x - self.last_state[0]) / self.dt
        vy = (y - self.last_state[1]) / self.dt
        v = np.hypot(vx, vy)
        # calc rate of turn
        w = (yaw - self.last_state[2]) / self.dt
        # calc acceleration
        a = (v - self.last_state[3]) / self.dt
        self.state = np.array([x, y, yaw, v, w, a])

    def read_state(self, qlabs: QuanserInteractiveLabs) -> None:
        """
        Read the state of the actor in the Quanser Interactive Labs environment.

        Parameters:
        - qlabs: QuanserInteractiveLabs: The Quanser Interactive Labs object
        """
        self.get_position(qlabs)
        self.cal_motion()
        self.last_state = self.state

    get_state = read_state


class VirtualRuningGear:
    """
    VirtualRuningGear is a class that represents the running gear of the QCar.

    Attributes:
    - class_id: int: The class id of the actor
    - actor_number: int: The actor number
    - throttle_coeff: float: The throttle coefficient
    - steering_coeff: float: The steering coefficient
    - FCN_QCAR_SET_VELOCITY_AND_REQUEST_STATE: int: The function code for setting velocity and 
        requesting state
    - FCN_QCAR_VELOCITY_STATE_RESPONSE: int: The function code for velocity state response
    """
    
    FCN_QCAR_SET_VELOCITY_AND_REQUEST_STATE = 10
    FCN_QCAR_VELOCITY_STATE_RESPONSE = 11

    def __init__(self, class_id: int, actor_number: int, throttle_coeff: float = 7.5) -> None:
        """
        Initializes the VirtualRuningGear object, which represents the running gear of the QCar, 
        usually use this to control the surrounding QCar.

        Parameters:
        - class_id: int: The class id of the QCar
        - actor_number: int: The actor number
        """
        self.class_id: int = class_id
        self.throttle_coeff: float = throttle_coeff
        self.actor_number: int = actor_number

    def read_write_std(
        self,
        qlabs: QuanserInteractiveLabs,
        throttle: float,
        steering: float,
        leds: list = [0, 0, 0, 0, 0]
    ) -> tuple:
        """
        Execute the action on the QCar based on the given throttle, steering, and LEDs.

        Parameters:
        - qlabs: QuanserInteractiveLabs: The Quanser Interactive Labs object
        - throttle: float: The throttle value
        - steering: float: The steering value
        - leds: list: The LEDs values

        Returns:
        - tuple: The location, rotation, frontHit, and rearHit
        """
        c = CommModularContainer()
        c.classID = self.class_id
        c.actorNumber = self.actor_number
        c.actorFunction = self.FCN_QCAR_SET_VELOCITY_AND_REQUEST_STATE
        c.payload = bytearray(struct.pack(">ffBBBBB", throttle * self.throttle_coeff, -steering, leds[0], leds[1], leds[2], leds[3], leds[4]))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        location = [0,0,0]
        rotation = [0,0,0]
        frontHit = False
        rearHit = False

        qlabs.flush_receive()

        if (qlabs.send_container(c)):
            c = qlabs.wait_for_container(self.class_id, self.actor_number, self.FCN_QCAR_VELOCITY_STATE_RESPONSE)

            if (c == None):
                print("No response from QCar")
                return False, location, rotation, frontHit, rearHit

            if len(c.payload) == 26:
                location[0], location[1], location[2], rotation[0], rotation[1], rotation[2], frontHit, rearHit, = struct.unpack(">ffffff??", c.payload[0:26])
                return True, location, rotation, frontHit, rearHit
            else:
                print("Invalid response from QCar")
                return False, location, rotation, frontHit, rearHit
        else:
            print("Failed to send container")
            return False, location, rotation, frontHit, rearHit
